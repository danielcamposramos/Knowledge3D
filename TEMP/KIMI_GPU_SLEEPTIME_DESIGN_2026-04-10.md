```markdown
# ARC3 GPU-Native Micro-Sleeptime Specification
**Version**: 1.0  
**System**: Knowledge3D Codex Implementation  
**Target**: ARC3 Episode Galaxy Consolidation Path  
**Grounding**: SLEEPTIME_PROTOCOL_SPECIFICATION.md (v2.0), FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md, SOVEREIGN_NSI_SPECIFICATION.md

---

## 1. Architectural Overview

The ARC3 agent implements a **dual-phase consolidation strategy** mapped to the SLEEPTIME 6-step protocol:

| Phase | Trigger | Kernel | Latency Budget | Purpose |
|-------|---------|--------|----------------|---------|
| **Micro-Sleeptime** | Between game steps (API wait) | `sleep_time_micro.ptx` | <2ms | Update rule confidences, flag stale rules, reinforce routes |
| **Attempt-Transition** | Between attempts (life lost/reset) | `sleep_cluster_refiner.ptx` + `galaxy_memory_updater.cu` + `lora_gpu.cu` | <50ms | Deep crystallization, Grammar Galaxy persist, specialist LoRA updates |

**Sovereignty Constraint**: All consolidation executes **inline** on the living KV instance. No process spawn, no VRAM unload, no checkpoint-and-reload between query and sleep phases.

---

## 2. Data Structures

### 2.1 VRAM Layout for ARC3 Sleep Context

```cpp
// C-aligned structures for PTX kernel arguments
// Located in: knowledge3d/knowledgeverse/arc3_sleep_structs.h

typedef struct {
    uint32_t color;           // Color value (0-15)
    uint32_t outcome_code;    // Encoded outcome (blocked=0, moved=1, death=2, goal=3)
    float confidence_delta;   // +strengthen, -weaken
    uint64_t timestamp;       // Frame counter
    uint8_t action_idx;       // 0-6 mapped action
} ARC3OutcomeEntry;          // 24 bytes

typedef struct {
    uint32_t rule_id;
    uint8_t condition_color;  // Color involved in rule
    uint8_t action_idx;
    float confidence;         // 0.0-1.0, prune <0.1
    uint32_t success_count;
    uint32_t failure_count;
    uint64_t last_activation;
    uint8_t flags;            // Bit 0: is_stale, Bit 1: is_blocked_path, Bit 2: is_open_path
} ARC3RuleEntry;             // 32 bytes

typedef struct {
    uint32_t object_id;
    uint8_t color;
    uint8_t semantic_class;   // 0=unknown, 1=avatar, 2=wall, 3=goal, 4=hazard
    float motion_variance;    // For avatar detection
    uint32_t death_assoc;     // Count of co-occurrence with death
    uint32_t goal_assoc;      // Count of co-occurrence with level_complete
    float bounding_box[4];    // Min/max row/col
} ARC3ObjectHypothesis;      // 40 bytes
```

### 2.2 Python Bridge Structures

```python
# knowledge3d/knowledgeverse/arc3_sleep_bindings.py
import ctypes
from dataclasses import dataclass
from typing import Optional, List

class MicroSleepArgs(ctypes.Structure):
    """Arguments for sleep_time_micro.ptx"""
    _fields_ = [
        ("outcome_ring_gpu", ctypes.c_void_p),      # Circular buffer of ARC3OutcomeEntry
        ("ring_head", ctypes.c_uint32),
        ("ring_capacity", ctypes.c_uint32),
        ("rule_table_gpu", ctypes.c_void_p),        # Mutable ARC3RuleEntry array
        ("rule_count", ctypes.c_uint32),
        ("object_hypotheses_gpu", ctypes.c_void_p), # ARC3ObjectHypothesis array
        ("hypothesis_count", ctypes.c_uint32),
        ("specialist_routes_gpu", ctypes.c_void_p), # Float array for route weights
        ("stale_flags_out_gpu", ctypes.c_void_p),   # Bitmask output for stale rules
        ("timestamp", ctypes.c_uint64),
        ("attempt_id", ctypes.c_uint32),
    ]

class AttemptTransitionArgs(ctypes.Structure):
    """Arguments for between-attempt deep consolidation"""
    _fields_ = [
        ("episode_rules_gpu", ctypes.c_void_p),     # Rules to evaluate for persistence
        ("rule_count", ctypes.c_uint32),
        ("brief_accumulator_gpu", ctypes.c_void_p), # Briefs from completed attempt
        ("brief_count", ctypes.c_uint32),
        ("grammar_galaxy_ptr", ctypes.c_void_p),    # Destination for Layer 2 stars
        ("specialist_lora_ptr", ctypes.c_void_p),     # LoRA weight buffer
        ("persistence_threshold", ctypes.c_float),  # Confidence threshold for promotion
        ("flags", ctypes.c_uint32),                  # Bit 0: purge_ephemeral
    ]

@dataclass
class ARC3SleepContext:
    """Python-side handle for GPU sleep state"""
    episode_id: str
    attempt_number: int
    outcome_ring: 'VRAMLessonRing'  # Reuse lesson ring for outcomes
    rule_buffer: 'VRAMTaskBuffer'   # Mutable rule storage
    object_buffer: 'VRAMTaskBuffer' # Object hypothesis storage
    stream: Optional['cuda.Stream'] = None  # CUDA stream for async execution
    
    def __post_init__(self):
        if self.stream is None:
            self.stream = cuda.Stream(non_blocking=True)
```

---

## 3. Between-Step Micro-Sleeptime (GPU-Native)

### 3.1 Function Signature

```python
# knowledge3d/knowledgeverse/arc3_episode_galaxy.py

def dispatch_micro_sleeptime_async(
    self,
    recent_outcomes: List[Dict[str, Any]],
    sleep_compute: 'SleepTimeComputeBridge',  # From sleep_time_compute.py
) -> 'cuda.Event':
    """
    Launch sleep_time_micro.ptx asynchronously while waiting for game API.
    
    Args:
        recent_outcomes: Last N outcomes (typically 5-10 steps)
        sleep_compute: Bridge to PTX runtime
        
    Returns:
        CUDA event to synchronize on before processing next frame
    """
    
def synchronize_micro_sleeptime(
    self,
    event: 'cuda.Event',
    timeout_ms: float = 100.0
) -> MicroSleepResults:
    """
    Block until GPU micro-sleeptime completes. Called after API response.
    Returns updated rule confidences and stale rule flags.
    """
```

### 3.2 Dispatch Sequence

```python
def run_micro_sleeptime_gpu(self, outcomes: list):
    # 1. Prepare GPU arguments
    args = MicroSleepArgs()
    args.outcome_ring_gpu = self.outcome_ring.gpu_ptr
    args.ring_head = self.outcome_ring.head
    args.rule_table_gpu = self.rule_buffer.input_buffer
    args.rule_count = len(self.active_rules)
    args.timestamp = get_frame_counter()
    
    # 2. Launch sleep_time_micro.ptx via bridge
    # Kernel performs:
    # - Parallel scan of outcomes to update rule confidences
    # - Stochastic gradient update on specialist route weights
    # - Atomic flagging of rules with confidence < PRUNE_THRESHOLD
    self.sleep_bridge.launch_async(
        kernel="micro_consolidate",
        args=args,
        stream=self.sleep_stream,
        block_size=256,
        grid_size=(args.rule_count + 255) // 256
    )
    
    # 3. Return immediately; CPU proceeds to game API call
    return self.sleep_stream.record_event()
```

### 3.3 PTX Kernel Logic (sleep_time_micro.ptx)

The kernel implements **Step 3 (Pattern Matching)** and **Step 4 (Confidence Update)** of the SLEEPTIME protocol:

```cuda
// Pseudocode for sleep_time_micro.ptx
__global__ void micro_consolidate(MicroSleepArgs args) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= args.rule_count) return;
    
    ARC3RuleEntry rule = args.rule_table_gpu[tid];
    float confidence_delta = 0.0f;
    int activation_count = 0;
    
    // Parallel scan recent outcomes (shared memory cooperative groups)
    for (int i = 0; i < args.ring_capacity; i++) {
        ARC3OutcomeEntry outcome = args.outcome_ring_gpu[i];
        
        // Pattern match: does outcome match rule condition?
        if (outcome.color == rule.condition_color && 
            outcome.action_idx == rule.action_idx) {
            
            activation_count++;
            
            // Update confidence based on outcome valence
            if (outcome.outcome_code == EXPECTED_OUTCOME(rule)) {
                confidence_delta += outcome.confidence_delta * 0.1f; // Hebbian strengthen
            } else {
                confidence_delta -= outcome.confidence_delta * 0.2f; // Anti-Hebbian weaken
            }
        }
    }
    
    // Apply updates with atomicity
    rule.confidence = clamp(rule.confidence + confidence_delta, 0.0f, 1.0f);
    rule.last_activation = args.timestamp;
    
    // Flag stale rules (Step 5: Pruning)
    if (rule.confidence < 0.1f || 
        (args.timestamp - rule.last_activation) > STALE_TIMEOUT) {
        rule.flags |= FLAG_STALE;
    }
    
    // Update rule table
    args.rule_table_gpu[tid] = rule;
    
    // Write stale flags for CPU-side cleanup
    if (rule.flags & FLAG_STALE) {
        atomicOr((unsigned int*)args.stale_flags_out_gpu, (1u << (tid % 32)));
    }
}
```

### 3.4 Integration Point

Replace the existing `run_micro_sleeptime` in `arc3_episode_galaxy.py`:

```python
# OLD (Python ThreadPool):
# with ThreadPoolExecutor(max_workers=3) as executor:
#     futures = [executor.submit(self._crystallize_rules), ...]

# NEW (GPU Native):
def run_micro_sleeptime(self, outcomes: list):
    """GPU-native micro-sleeptime while waiting for API."""
    if not outcomes:
        return
    
    # Launch async GPU work
    event = self.dispatch_micro_sleeptime_async(outcomes, self.sleep_bridge)
    
    # CPU: Fire game API request (network I/O, overlaps with GPU)
    api_response = self.game_client.query_action_async()
    
    # Synchronize: Wait for both GPU and API
    api_result = api_response.get(timeout=5.0)
    self.synchronize_micro_sleeptime(event, timeout_ms=100)
    
    # CPU: Cleanup stale rules flagged by GPU
    self._prune_stale_rules_from_gpu_flags()
    
    return api_result
```

---

## 4. Between-Attempt Consolidation (GPU-Native)

### 4.1 State Transition Specification

**Attempt Transition Invariants** (per SOVEREIGN_NSI_SPECIFICATION.md):

| VRAM Region | Action | Justification |
|-------------|--------|---------------|
| Grammar Galaxy (Layer 2) | **PERSIST** | Permanent semantic knowledge (walls, goals) |
| Episode Rules (blocked/open paths) | **PERSIST** | Learned physics of current level |
| Object Identity Stars | **PERSIST** | Recognized avatars/hazards from prior attempts |
| Action History Ring | **PURGE** | Attempt-specific trajectory, not reusable |
| Frame Cache (last 4 grids) | **PURGE** | Ephemeral state, new attempt starts fresh |
| Drift/Stuck Signals | **PURGE** | Local minima detectors, not transferable |
| Specialist LoRA Weights | **UPDATE** | Contrastive learning from attempt outcome |

### 4.2 Function Signature

```python
def consolidate_between_attempts(
    self,
    attempt_result: AttemptResult,  # success, failure, timeout
    cluster_refiner: 'SleepClusterRefinerBridge',  # From sleep_cluster_kernels.py
    memory_updater: 'GalaxyMemoryUpdater',         # galaxy_memory_updater.cu wrapper
    lora_gpu: 'LoRAGPU',                          # lora_gpu.cu wrapper
) -> ConsolidationReceipt:
    """
    Deep consolidation between attempts. Blocking call.
    Must complete before next attempt begins.
    """
```

### 4.3 Three-Kernel Dispatch Sequence

```python
def consolidate_between_attempts_gpu(self, attempt_result: AttemptResult):
    # Prepare transition context
    transition = AttemptTransitionArgs()
    transition.episode_rules_gpu = self.rule_buffer.input_buffer
    transition.rule_count = self.active_rule_count
    transition.brief_accumulator_gpu = self.lesson_ring.gpu_ptr
    transition.brief_count = self.lesson_ring.count
    transition.grammar_galaxy_ptr = self.grammar_star_table.gpu_ptr
    transition.persistence_threshold = 0.7  # Promote rules >70% confidence
    transition.flags = 0x1 if attempt_result.is_terminal else 0x0
    
    # KERNEL 1: Deep Crystallization (sleep_cluster_refiner.ptx)
    # - Clusters co-activated rules from the attempt
    # - Identifies stable paths (high co-activation = crystallized route)
    self.cluster_refiner.launch(
        kernel="refine_attempt_clusters",
        args=transition,
        grid=(1, 1, 1),  # Single block for atomic cluster ops
        block=(256, 1, 1)
    )
    
    # KERNEL 2: Grammar Galaxy Persistence (galaxy_memory_updater.cu)
    # - Scores rules via contrastive learning (success vs failure trajectories)
    # - Persists strong rules (>threshold) to Layer 2 as Meaning Stars
    # - Updates star provenance with attempt_id
    self.memory_updater.launch(
        operation="persist_strong_rules",
        source_buffer=self.rule_buffer.input_buffer,
        destination_galaxy=self.grammar_star_table,
        threshold=transition.persistence_threshold,
        attempt_id=self.current_attempt_id
    )
    
    # KERNEL 3: Specialist Weight Update (lora_gpu.cu)
    # - Contrastive update: successful attempt weights vs shadow copy
    # - Strengthens routes that led to goal, weakens dead-end paths
    self.lora_gpu.update_weights_contrastive(
        episode_buffer=self.episode_vram_buffer,
        outcome=attempt_result.outcome_type,  # "success" | "failure" | "timeout"
        learning_rate=0.001
    )
    
    # Synchronize all kernels
    cuda.synchronize()
    
    # CPU-side cleanup post-GPU
    self._purge_ephemeral_state()  # Clear action ring, frame cache
    
    return ConsolidationReceipt(
        rules_persisted=self.memory_updater.get_persisted_count(),
        clusters_refinined=self.cluster_refiner.get_cluster_delta(),
        lora_delta_norm=self.lora_gpu.get_update_norm(),
        timestamp=now()
    )
```

### 4.4 VRAM Persistence Protocol

The `galaxy_memory_updater.cu` kernel implements **Layer 2 to Layer 3 transfer**:

```cuda
// galaxy_memory_updater.cu logic
__global__ void update_galaxy_memory(GalaxyUpdateArgs args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.rule_count) return;
    
    ARC3RuleEntry rule = args.rules[idx];
    
    // Check if rule qualifies for promotion to Grammar Galaxy
    if (rule.confidence > args.persistence_threshold && 
        !(rule.flags & FLAG_STALE)) {
        
        // Create or update Meaning Star in Grammar Galaxy
        MeaningStar star;
        star.star_id = hash_rule(rule);
        star.semantic_type = infer_semantic_type(rule); // blocked_path, goal_region, etc.
        star.vram_ptr = args.destination_galaxy.allocate_slot();
        star.provenance = args.attempt_id;
        star.confidence = rule.confidence;
        
        // Atomic write to Grammar Galaxy (Layer 2 permanent)
        args.destination_galaxy.write_atomic(star);
        
        // Mark as crystallized in episode buffer
        rule.flags |= FLAG_CRYSTALLIZED;
        args.rules[idx] = rule;
    }
}
```

---

## 5. Character Identification Protocol

### 5.1 Algorithm: Semantic Inference via Action Correlation

**Input**: Sequence of `(grid, action, outcome)` tuples from single attempt  
**Output**: Galaxy stars with `semantic_class` assigned (avatar, wall, goal, hazard)

**GPU Kernel Steps**:

1. **Motion Correlation Kernel** (parallel per color):
   ```cuda
   for each color c in [0, 15]:
       if action causes centroid(c) to move:
           motion_score[c] += correlation(action_delta, centroid_delta)
       else:
           motion_score[c] *= decay_factor
   ```

2. **Outcome Association Kernel** (parallel per observation):
   ```cuda
   if outcome == "death":
       for each color c in grid: hazard_assoc[c]++
   if outcome == "level_complete":
       for each color c in grid: goal_assoc[c]++
   if outcome == "blocked":
       blocking_color = grid[action_direction]; wall_assoc[blocking_color]++
   ```

3. **Consensus Classification**:
   ```cuda
   if motion_score[c] > MOTION_THRESHOLD:
       semantic_class = AVATAR
   else if hazard_assoc[c] > DEATH_THRESHOLD:
       semantic_class = HAZARD
   else if goal_assoc[c] > GOAL_THRESHOLD:
       semantic_class = GOAL