# ARC3 Between-Attempt Consolidation Architecture Review

## Executive Summary

**CRITICAL SOVEREIGNTY VIOLATION DETECTED**: The current Python ThreadPoolExecutor implementation for `_crystallize_rules`, `_reinforce_routes`, `_classify_objects` directly violates SLEEPTIME_PROTOCOL_SPECIFICATION.md Section 0.1. Consolidation MUST execute on GPU via PTX kernels, not CPU-bound Python threads.

---

## 1. SOVEREIGNTY COMPLIANCE — VIOLATION FLAGGED 🚩

### Current State (VIOLATION)
```python
# CURRENT: CPU-bound ThreadPoolExecutor (SOVEREIGNTY VIOLATION)
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.submit(_crystallize_rules, galaxy_state)
    executor.submit(_reinforce_routes, route_table)
    executor.submit(_classify_objects, perception_buffer)
```

### Spec Reference: SLEEPTIME_PROTOCOL_SPECIFICATION.md §0.1

> **"Consolidation MUST execute on GPU via PTX kernels.** The Python entry point (`jarvis_sleep_consolidation()`) LAUNCHES GPU kernels — it does NOT do the work in Python."

> **"GPU utilization during sleep MUST be visible (>0% SM occupancy).** If consolidation runs entirely on CPU with idle GPU, it is a sovereignty violation."

### Required Architecture (COMPLIANT)

```python
# CORRECT: Python launches GPU kernels, does NOT do the work
def jarvis_sleep_consolidation(galaxy_state, briefs_pending):
    """
    S0.1 COMPLIANT: Python is launcher only, GPU does the work
    """
    # 1. Launch micro-consolidation kernel
    sleep_time_micro.ptx.launch(
        grid_dims=(256, 1, 1),
        block_dims=(32, 1, 1),
        args=[galaxy_state.device_ptr, briefs_pending.device_ptr, consolidation_config]
    )
    
    # 2. Launch cluster refinement kernel
    sleep_cluster_refiner.ptx.launch(
        grid_dims=(128, 1, 1),
        block_dims=(64, 1, 1),
        args=[galaxy_state.clusters, coactivation_matrix, refinement_params]
    )
    
    # 3. Update galaxy memory scores (CU kernel)
    galaxy_memory_updater.cu.launch(
        grid_dims=(512, 1, 1),
        block_dims=(32, 1, 1),
        args=[galaxy_state.entries, score_deltas, decay_factors]
    )
    
    # 4. Specialist weight updates via LoRA (CU kernel)
    lora_gpu.cu.launch(
        grid_dims=(256, 1, 1),
        block_dims=(64, 1, 1),
        args=[specialist_weights, shadow_copy, contrastive_gradients]
    )
    
    # 5. Sync and verify (Python waits for GPU completion)
    torch.cuda.synchronize()
    verify_consolidation_checksums(galaxy_state)
```

### Kernel Wiring for ARC3 Between-Step vs Between-Attempt

| Consolidation Type | Kernels to Invoke | Trigger Condition | VRAM Impact |
|-------------------|-------------------|-------------------|-------------|
| **Between-Step** (micro-sleep) | `sleep_time_micro.ptx` only | Every N steps (default: 10) | Minimal - score updates only |
| **Between-Attempt** (full sleep) | All 4 kernels + `galaxy_memory_updater.cu` + `lora_gpu.cu` | Attempt complete, before next attempt | Full - cluster refinement + weight updates |
| **Between-Game** (deep sleep) | All kernels + House checkpoint | Game complete | Full + persistent storage write |

---

## 2. GAME KNOWLEDGE PERSISTENCE — 4-LAYER ARCHITECTURE

### Spec Reference: FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §1.2-1.5

```
Layer 4: META-RULES (Strategy/Eloquence) — when/why to apply
    ↓
Layer 3: RULES (Grammar/Transformation) — how to transform
    ↓
Layer 2: MEANING (Words/Semantics) — what it means
    ↓
Layer 1: FORM (Characters/Glyphs) — how it looks
```

### ARC3 Knowledge Mapping

| Knowledge Type | Layer | Galaxy Star Name (Meaning-Based) | Storage Format |
|---------------|-------|----------------------------------|----------------|
| "color 3 = avatar" | **Layer 2** | `spatial_grid_agent_marker` | WordDefinition with color_coefficient=3 |
| "color 5 = wall" | **Layer 2** | `spatial_grid_obstacle_marker` | WordDefinition with color_coefficient=5 |
| "adjacent to color 5 → ACTION2 blocked" | **Layer 3** | `spatial_adjacency_block_rule` | RPN program with condition → action_mask |
| "prefer untested over known-blocked" | **Layer 4** | `exploration_priority_strategy` | Meta-rule with uncertainty_weight > block_weight |

### Galaxy Star Structure (Layer 2 Example)

```python
@dataclass
class WordDefinition:
    word_id: str                    # "spatial_grid_agent_marker"
    layer: int                      # 2 (MEANING)
    semantic_vector: np.float32[768]
    references: List[StarReference] # Links to Layer 1 glyphs
    coefficients: Dict[str, float]  # {"color": 3.0, "movable": 1.0, "self": 1.0}
    provenance: str                 # "arc3_episode_001_attempt_003"
    consolidation_count: int        # How many sleep cycles strengthened this
    last_accessed: int              # Timestamp for decay calculation
```

### Critical Naming Convention (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §0)

**VIOLATION TO AVOID**: Do NOT name stars `arc3_avatar_color` or `arc3_wall_rule`

**CORRECT**: `spatial_grid_agent_marker`, `spatial_grid_obstacle_marker`, `spatial_adjacency_block_rule`

> **"All stars, Galaxy entries, methods, symlinks, and knowledge artifacts MUST be named by their MEANING — never by the benchmark or dataset that motivated their creation."**

---

## 3. BETWEEN-ATTEMPT CONTRACT — INLINE EXECUTION

### Spec Reference: SLEEPTIME_PROTOCOL_SPECIFICATION.md §0.3

> **"Consolidation runs on the same KV instance that processed the queries — while stars, briefs, and specialist weights are still loaded in VRAM."**

> **"It does NOT: Spawn a separate process, Unload the system and reload for consolidation, Require a restart between query phase and sleep phase"**

### State Persistence Matrix

| State Component | Between-Step | Between-Attempt | Between-Game | Spec Reference |
|-----------------|--------------|-----------------|--------------|----------------|
| **Knowledgeverse Instance** | ✅ Persist | ✅ Persist | ✅ Persist | S0.3 Inline Execution |
| **VRAM Allocation** | ✅ Persist | ✅ Persist | ⚠️ Optional checkpoint | S0.3 |
| **Star Table (Galaxy)** | ✅ Persist | ✅ Persist | ✅ Persist + House | §2.1 State Machine |
| **Episode Galaxy (Learned Rules)** | ✅ Persist | ✅ Persist | ✅ Persist | §1.1 Biological Fidelity |
| **Attempt-Specific State** | ✅ Persist | ❌ Purge | ❌ Purge | Daniel's Contract |
| **Action Ring Buffer** | ✅ Persist | ❌ Reset | ❌ Reset | Transient State |
| **Frame Buffer** | ✅ Persist | ❌ Reset | ❌ Reset | Transient State |
| **Stuck Signals** | ✅ Persist | ❌ Reset | ❌ Reset | Transient State |
| **Brief Queue** | ✅ Accumulate | ✅ Consolidate | ✅ Consolidate All | §0.2 Trigger Conditions |

### Implementation Pattern

```python
class ARC3Agent:
    def __init__(self):
        self.knowledgeverse = Knowledgeverse()  # Single instance, never reloaded
        self.episode_galaxy = EpisodeGalaxy()   # Persists across attempts
        self.transient_state = TransientState() # Resets per attempt
        
    def complete_attempt(self, success: bool):
        # 1. Consolidate learned knowledge to Episode Galaxy (GPU)
        self._gpu_consolidate_attempt_knowledge()
        
        # 2. Purge ONLY transient state (NOT Episode Galaxy)
        self.transient_state.reset()
        
        # 3. Keep Knowledgeverse, VRAM, Star Table intact
        # NO reload, NO checkpoint save/load cycle
        
    def _gpu_consolidate_attempt_knowledge(self):
        # S0.1 COMPLIANT: Launch PTX kernels, don't do work in Python
        sleep_cluster_refiner.ptx.launch(...)
        galaxy_memory_updater.cu.launch(...)
        # ... (see Section 1)
```

---

## 4. VECTORDOTMAP INTEGRATION — GRID AS DOT FIELD

### Conceptual Mapping

The ARC3 grid IS a VectorDotMap field:
- Each cell = a dot at position (x, y)
- Color = coefficient value (or additional dimension)
- State (explored/blocked/visited) = additional coefficients

### VectorDotMap Field Coefficient Schema

```python
@dataclass
class GridDotField:
    """
    Each grid cell is a dot with multi-dimensional coefficients
    Persists between attempts as part of Episode Galaxy spatial memory
    """
    position: np.float32[2]      # (x, y) grid coordinates
    color_coefficient: float     # 0-9 for ARC3 colors
    exploration_coefficient: float  # 0.0 (unvisited) → 1.0 (fully explored)
    block_probability: float     # 0.0 (free) → 1.0 (confirmed wall)
    visit_timestamp: float       # Last access time for decay
    action_success_history: np.float32[4]  # Per-action success rates
```

### GPU Kernel for Dot Field Consolidation

```cuda
// galaxy_memory_updater.cu — Dot field coefficient updates
__global__ void update_grid_dot_field(
    GridDotField* dots,
    float* action_outcomes,
    int num_dots,
    float learning_rate,
    float decay_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dots) return;
    
    GridDotField& dot = dots[idx];
    
    // Update block probability based on action outcomes
    if (action_outcomes[idx] < 0) {
        dot.block_probability = fminf(1.0f, dot.block_probability + learning_rate);
    } else {
        dot.block_probability = fmaxf(0.0f, dot.block_probability - learning_rate * 0.5f);
    }
    
    // Exploration coefficient increases with visits
    dot.exploration_coefficient = fminf(1.0f, dot.exploration_coefficient + 0.1f);
    
    // Time-based decay for stale information
    dot.exploration_coefficient *= decay_factor;
}
```

### Persistence Between Attempts

```python
class EpisodeGalaxy:
    def __init__(self):
        self.spatial_dot_field = GridDotFieldGPU()  # VRAM-resident
        self.learned_rules = RuleGalaxy()           # Layer 3+4 stars
        self.semantic_mappings = MeaningGalaxy()    # Layer 2 stars
        
    def between_attempt_persist(self):
        # Keep all of the above in VRAM
        # Only clear transient buffers
        pass
        
    def between_attempt_reset(self):
        # DO NOT clear spatial_dot_field
        # DO NOT clear learned_rules
        # DO NOT clear semantic_mappings
        self.transient_action_buffer.clear()
        self.current_frame_buffer.clear()
        self.stuck_detector.reset()
```

---

## 5. SPECIFIC VIOLATIONS TO REMEDIATE

| Violation | Current Implementation | Required Fix | Spec Reference |
|-----------|----------------------|--------------|----------------|
| **CPU Consolidation** | ThreadPoolExecutor with Python functions | Launch PTX/CU kernels from Python | S0.1 |
| **Benchmark Naming** | Potential `arc3_*` star names | Use `spatial_grid_*` meaning-based names | Foundational §0 |
| **System Reload** | Potential checkpoint save/load between attempts | Inline execution, same KV instance | S0.3 |
| **Full State Reset** | Potential Episode Galaxy clear between attempts | Persist Episode Galaxy, reset only transient | Daniel's Contract |
| **Grid as Image** | Potential frame buffer as RGB image | Grid as VectorDotMap with coefficients | VectorDotMap Integration |

---

## 6. RECOMMENDED ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGEVERSE (Single Instance)              │
│                    VRAM-Resident, Never Reloads                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Episode Galaxy │    │  Meaning Galaxy │    │  Rule Galaxy│  │
│  │  (Persists)     │    │  (Layer 2)      │    │  (Layer 3-4)│  │
│  │  - Dot Field    │    │  - color=avatar │    │  - adjacency│  │
│  │  - Learned Rules│    │  - color=wall   │    │  - priority │  │
│  └────────┬────────┘    └────────┬────────┘    └──────┬──────┘  │
│           │                      │                     │         │
│           └──────────────────────┼─────────────────────┘         │
│                                  │                                │
│                    ┌─────────────▼─────────────┐                  │
│                    │   GPU Consolidation       │                  │
│                    │   (PTX/CU Kernels)        │                  │
│                    │   - sleep_time_micro.ptx  │                  │
│                    │   - sleep_cluster_refiner │                  │
│                    │   - galaxy_memory_updater │                  │
│                    │   - lora_gpu.cu           │                  │
│                    └─────────────┬─────────────┘                  │
│                                  │                                │
│  ┌───────────────────────────────▼─────────────────────────────┐ │
│  │              Transient State (Resets Per Attempt)            │ │
│  │  - Action Ring Buffer  - Frame Buffer  - Stuck Signals      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. ACTION ITEMS

1. **IMMEDIATE**: Replace ThreadPoolExecutor consolidation with PTX kernel launches
2. **HIGH**: Audit all star names for benchmark-specific naming violations
3. **HIGH**: Verify Episode Galaxy persists in VRAM between attempts (no unload/reload)
4. **MEDIUM**: Implement VectorDotMap grid representation with coefficient schema
5. **MEDIUM**: Add GPU occupancy monitoring to verify >0% SM utilization during sleep
6. **LOW**: Document consolidation trigger conditions per S0.2 (idle timeout, brief batch, memory pressure)

---

## 8. COMPLIANCE CHECKLIST

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPU-native consolidation (S0.1) | ❌ VIOLATION | Current: ThreadPoolExecutor |
| Inline execution (S0.3) | ⚠️ VERIFY | Confirm no reload between attempts |
| Meaning-based naming (Foundational §0) | ⚠️ VERIFY | Audit star names |
| Episode Galaxy persistence | ⚠️ VERIFY | Confirm VRAM retention |
| Transient state isolation | ⚠️ VERIFY | Confirm reset scope |
| VectorDotMap grid encoding | ❌ NOT IMPLEMENTED | Current: likely image buffer |

**Overall Compliance: 33% (2/6 requirements met)**

**Priority: CRITICAL** — Sovereignty violation must be remediated before production deployment.