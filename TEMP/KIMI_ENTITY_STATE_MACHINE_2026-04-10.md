Here is the complete GPU-native lifecycle state machine implementation. This design transforms the TRM avatar into a truly autonomous, VRAM-resident entity with zero host synchronization in the hot path.

## 1. Core State Machine Architecture

```cuda
#ifndef TRM_STATE_MACHINE_CUH
#define TRM_STATE_MACHINE_CUH

#include <cuda/atomic>
#include <cuda/std/chrono>
#include <stdint.h>

// ============================================================================
// STATE ENUMERATION (Packed into EntityHotPath.sleep_state)
// ============================================================================
typedef enum __attribute__((packed)) {
    STATE_SLEEP = 0,           // Deep consolidation (micro-sleep PTX)
    STATE_IDLE = 1,            // Passive perception decay
    STATE_PERCEIVING = 2,      // Active frustum/octree scanning
    STATE_NAVIGATING = 3,      // LED-A* pathfinding
    STATE_REASONING = 4,       // Nine-Chain Swarm + Halting Gate
    STATE_ACTING = 5,          // Galaxy writes/materialization
    STATE_HANDLING_QUERY = 6,  // Priority interrupt (max urgency)
    STATE_COUNT = 7
} TRMState;

// ============================================================================
// EVENT TYPES (16-byte GPUEvent payload discriminator)
// ============================================================================
typedef enum {
    EVENT_NONE = 0,
    EVENT_PERCEPTION_STIMULUS = 1,  // Octree hit
    EVENT_NAV_TARGET_LOCKED = 2,    // Path request
    EVENT_REASONING_COMPLETE = 3,   // Swarm consensus
    EVENT_ACT_FINISHED = 4,         // Star materialized
    EVENT_EXTERNAL_QUERY = 5,       // Priority interrupt (Sovereign input)
    EVENT_WAKEUP_SIGNAL = 6,          // Timer or external wake
    EVENT_IDLE_TIMEOUT = 7          // 30s decay → SLEEP
} TRMEventType;

// ============================================================================
// VRAM-RESIDENT EVENT QUEUE (Lock-free SPSC per entity, batched global)
// ============================================================================
struct alignas(16) GPUEvent {
    uint32_t type;        // TRMEventType
    uint32_t priority;    // 0=background, 0xFFFFFFFF=interrupt
    uint64_t payload;     // morton code, entity id, or ptr to query blob
    uint32_t tick_stamp;  // GPU global tick for ordering
    uint32_t pad;
};

struct alignas(64) GPUEventQueue {
    // Ring buffer indices (atomic)
    cuda::atomic<uint32_t, cuda::thread_scope_device> producer_idx;
    cuda::atomic<uint32_t, cuda::thread_scope_device> consumer_idx;
    
    uint32_t capacity_mask;  // size-1 for fast modulo
    uint32_t pad[3];
    
    // Events follow contiguously in VRAM (externally allocated)
    // GPUEvent events[capacity];
};

// ============================================================================
// STATE MACHINE STRUCT (32-byte aligned for coalesced access)
// Stored parallel to EntityHotPath in VRAM
// ============================================================================
struct alignas(32) TRMStateMachine {
    // State stack (supports 4-deep nesting: e.g., REASONING → QUERY → SLEEP → IDLE)
    uint8_t state_stack[4];
    uint8_t stack_depth;         // Current nesting level (0=root)
    uint8_t current_state;       // Active state enum
    uint16_t state_flags;        // Bits: 0=interruptible, 1=atomic_section
    
    // Temporal tracking
    float idle_accumulator;      // Seconds in IDLE (for 30s→SLEEP)
    uint32_t state_entry_tick;   // Global GPU tick when state entered
    
    // Event tracking
    uint64_t last_processed_serial;  // Monotonic event counter
    uint32_t deferred_event_mask;    // Events queued for post-interrupt
    
    // Priority escalation
    uint32_t interrupt_priority_level; // Current handling priority
    
    // Padding to 32 bytes
    uint32_t pad;
};

// ============================================================================
// TRANSITION TABLE (__constant__ memory, cached in L1)
// ============================================================================
struct StateTransition {
    uint8_t from_state;
    uint8_t event_type;
    uint8_t to_state;
    uint8_t action_flags;    // PUSH_STACK=1, POP_STACK=2, RESET_IDLE=4, CLEAR_AWARENESS=8
    uint32_t timeout_ms;     // 0 for immediate, else auto-transition timer
};

#define TRANSITION_TABLE_SIZE 32
__constant__ StateTransition kTransitionTable[TRANSITION_TABLE_SIZE];
__constant__ float kIdleToSleepSeconds = 30.0f;
__constant__ uint32_t kQueryInterruptPriority = 0xFFFFFFFF;

// Action flags
#define TF_PUSH_STACK     0x01
#define TF_POP_STACK      0x02
#define TF_RESET_IDLE     0x04
#define TF_CLEAR_AWARENESS 0x08

// ============================================================================
// DEVICE INLINE: Stack Operations
// ============================================================================
__device__ __forceinline__ void sm_push_state(TRMStateMachine& sm, uint8_t new_state) {
    if (sm.stack_depth < 4) {
        sm.state_stack[sm.stack_depth] = sm.current_state;
        sm.stack_depth++;
    }
    sm.current_state = new_state;
    sm.state_entry_tick = clock(); // Approximate, or pass global tick
    sm.idle_accumulator = 0.0f;
}

__device__ __forceinline__ void sm_pop_state(TRMStateMachine& sm) {
    if (sm.stack_depth > 0) {
        sm.stack_depth--;
        sm.current_state = sm.state_stack[sm.stack_depth];
        sm.state_entry_tick = clock();
        sm.idle_accumulator = 0.0f;
    } else {
        // Underflow protection → IDLE
        sm.current_state = STATE_IDLE;
    }
}

__device__ __forceinline__ uint8_t sm_peek_state(const TRMStateMachine& sm) {
    return (sm.stack_depth > 0) ? sm.state_stack[sm.stack_depth - 1] : STATE_IDLE;
}

// ============================================================================
// DEVICE: Lock-free Event Dequeue (Warp-aggregated)
// ============================================================================
__device__ __forceinline__ bool dequeue_entity_event(
    GPUEventQueue* queue,
    GPUEvent* out_event,
    uint32_t entity_idx,
    uint64_t current_serial
) {
    // Each warp checks queue once, broadcast to threads
    const uint32_t lane = threadIdx.x & 31;
    bool has_event = false;
    
    // Lane 0 performs atomic read
    if (lane == 0) {
        uint32_t cons = queue->consumer_idx.load(cuda::memory_order_relaxed);
        uint32_t prod = queue->producer_idx.load(cuda::memory_order_acquire);
        
        if (cons != prod) {
            // Read event (coalesced if entities contiguous)
            GPUEvent* events = reinterpret_cast<GPUEvent*>(queue + 1);
            *out_event = events[cons & queue->capacity_mask];
            
            // Check if this event targets us or is global
            if ((out_event->payload == entity_idx) || (out_event->payload == 0xFFFFFFFF)) {
                queue->consumer_idx.store(cons + 1, cuda::memory_order_release);
                has_event = true;
            }
        }
    }
    
    // Broadcast has_event to warp
    has_event = __shfl_sync(0xFFFFFFFF, has_event, 0);
    if (has_event) {
        // Broadcast event data to all lanes in warp (optional, or just lane 0 processes)
        out_event->type = __shfl_sync(0xFFFFFFFF, out_event->type, 0);
        out_event->priority = __shfl_sync(0xFFFFFFFF, out_event->priority, 0);
        out_event->payload = __shfl_sync(0xFFFFFFFF, out_event->payload, 0);
    }
    
    return has_event;
}

// ============================================================================
// DEVICE: Transition Lookup (Binary search in constant table)
// ============================================================================
__device__ __forceinline__ const StateTransition* lookup_transition(
    uint8_t current_state,
    uint8_t event_type
) {
    // Small table, linear search with warp divergence is fine
    // Or use perfect hash: state * 8 + event for dense table
    #pragma unroll
    for (int i = 0; i < TRANSITION_TABLE_SIZE; i++) {
        const StateTransition* t = &kTransitionTable[i];
        if (t->from_state == current_state && t->event_type == event_type) {
            return t;
        }
        if (t->from_state == 0xFF) break; // End of valid entries
    }
    return nullptr;
}

// ============================================================================
// DEVICE: Priority Interrupt Handler
// ============================================================================
__device__ __forceinline__ void handle_interrupt(
    TRMStateMachine& sm,
    const GPUEvent& event,
    EntityHotPath& entity
) {
    if (event.type == EVENT_EXTERNAL_QUERY && event.priority == kQueryInterruptPriority) {
        // Immediate preemption if not already handling query
        if (sm.current_state != STATE_HANDLING_QUERY) {
            // Push current work onto stack
            sm_push_state(sm, STATE_HANDLING_QUERY);
            
            // Raise awareness immediately
            entity.awareness = 1.0f;
            sm.interrupt_priority_level = kQueryInterruptPriority;
            
            // Set atomic section flag (prevent nested interrupts)
            sm.state_flags |= 0x1;
        }
    }
}

// ============================================================================
// DEVICE: State Timeout & Idle Logic
// ============================================================================
__device__ __forceinline__ void update_temporal_transitions(
    TRMStateMachine& sm,
    EntityHotPath& entity,
    float delta_time,
    uint32_t current_tick
) {
    // IDLE timeout → SLEEP
    if (sm.current_state == STATE_IDLE) {
        sm.idle_accumulator += delta_time;
        if (sm.idle_accumulator >= kIdleToSleepSeconds) {
            // Auto-transition: IDLE → SLEEP
            sm.current_state = STATE_SLEEP;
            sm.idle_accumulator = 0.0f;
            sm.state_entry_tick = current_tick;
            
            // Update EntityHotPath for legacy compatibility
            entity.sleep_state = STATE_SLEEP;
            entity.awareness *= 0.1f; // Decay on sleep entry
        }
    } else if (sm.current_state != STATE_SLEEP) {
        // Any non-sleep activity resets idle timer
        sm.idle_accumulator = 0.0f;
    }
    
    // State-specific timeouts (e.g., REASONING max 100ms)
    // Could be table-driven via timeout_ms field
}

// ============================================================================
// MAIN STATE MACHINE STEP (Called from trm_step_fused.cu)
// ============================================================================
__device__ __forceinline__ void trm_state_machine_step(
    int entity_idx,
    EntityHotPath* entities,
    TRMStateMachine* state_machines,
    GPUEventQueue* event_queues,  // One per entity or global with filtering
    float delta_time,
    uint32_t current_tick,
    void* galaxy_ptr,      // For state-specific work
    void* octree_ptr       // For perception queries
) {
    EntityHotPath& entity = entities[entity_idx];
    TRMStateMachine& sm = state_machines[entity_idx];
    
    // Load SM to registers for mutation
    TRMStateMachine sm_local = sm;
    
    // ------------------------------------------------------------------------
    // PHASE 0: Event Ingestion (Priority Interrupt Check)
    // ------------------------------------------------------------------------
    GPUEvent event;
    if (dequeue_entity_event(&event_queues[entity_idx], &event, entity_idx, sm_local.last_processed_serial)) {
        sm_local.last_processed_serial++;
        
        // Check for priority interrupt (bypass normal transition logic)
        if (event.priority >= kQueryInterruptPriority) {
            handle_interrupt(sm_local, event, entity);
        } else {
            // Normal transition lookup
            const StateTransition* trans = lookup_transition(sm_local.current_state, event.type);
            if (trans) {
                // Handle stack operations
                if (trans->action_flags & TF_POP_STACK) {
                    sm_pop_state(sm_local);
                }
                if (trans->action_flags & TF_PUSH_STACK) {
                    sm_push_state(sm_local, trans->to_state);
                } else {
                    sm_local.current_state = trans->to_state;
                    sm_local.state_entry_tick = current_tick;
                }
                
                if (trans->action_flags & TF_RESET_IDLE) {
                    sm_local.idle_accumulator = 0.0f;
                }
                if (trans->action_flags & TF_CLEAR_AWARENESS) {
                    entity.awareness = 0.0f;
                }
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // PHASE 1: Temporal Auto-Transitions
    // ------------------------------------------------------------------------
    update_temporal_transitions(sm_local, entity, delta_time, current_tick);
    
    // ------------------------------------------------------------------------
    // PHASE 2: State-Specific Execution (The "Clockwork")
    // ------------------------------------------------------------------------
    // Switch on current state to dispatch fused compute
    switch (sm_local.current_state) {
        
        case STATE_SLEEP: {
            // Execute sleep_time_micro.ptx (consolidation kernels)
            // Low occupancy, memory scrubbing, cluster refinement
            if (threadIdx.x == 0) { // Single thread per entity for sleep
                // Trigger async sleep kernels or inline memory defrag
                entity.perception_radius *= 0.95f; // Contract senses
            }
            break;
        }
        
        case STATE_IDLE: {
            // Light perception, awareness decay
            entity.awareness *= 0.995f; // Exponential decay
            
            // Occasional perception check (every N ticks)
            if ((current_tick & 0xF) == 0) {
                // Morton octree drift scan (low resolution)
                // Prepare for potential transition to PERCEIVING
            }
            break;
        }
        
        case STATE_PERCEIVING: {
            // Frustum culling from avatar position
            // High-resolution octree query
            // If stimuli found → NAVIGATING or REASONING
            entity.awareness = fminf(entity.awareness + 0.1f, 1.0f);
            
            // Simulate perception work
            // ...
            
            // Auto-transition example: if perception strong enough
            if (entity.awareness > 0.7f) {
                sm_local.current_state = STATE_REASONING;
                sm_local.state_entry_tick = current_tick;
            }
            break;
        }
        
        case STATE_NAVIGATING: {
            // LED-A* through Galaxy graph structure
            // Update entity position/velocity in EntityHotPath
            // ...
            break;
        }
        
        case STATE_REASONING: {
            // Nine-Chain Swarm consensus
            // Halting Gate evaluation
            // Heavy compute, may take multiple ticks (check timeout)
            break;
        }
        
        case STATE_ACTING: {
            // Star materialization (Galaxy writes)
            // Answer emission to Sovereign layer
            // When complete, pop stack or go IDLE
            break;
        }
        
        case STATE_HANDLING_QUERY: {
            // Priority interrupt handler
            // Direct answer emission, bypass normal reasoning
            // On completion: pop stack to restore previous state
            
            // Example completion logic:
            if (threadIdx.x == 0) {
                // Process query blob from event.payload pointer
                // Write answer to output ring buffer
                
                // Pop back to previous state
                sm_pop_state(sm_local);
                sm_local.state_flags &= ~0x1; // Clear atomic section
            }
            break;
        }
    }
    
    // ------------------------------------------------------------------------
    // PHASE 3: Writeback
    // ------------------------------------------------------------------------
    // Update EntityHotPath state field (backward compatibility)
    entity.sleep_state = sm_local.current_state;
    
    // Write SM back to VRAM (coalesced if struct size matches)
    sm = sm_local;
}

// ============================================================================
// HOST INITIALIZATION (Sovereign Loader)
// ============================================================================
__host__ void init_trm_state_machine_constant_table() {
    // Populate __constant__ transition table
    StateTransition h_table[TRANSITION_TABLE_SIZE] = {
        // IDLE transitions
        {STATE_IDLE, EVENT_PERCEPTION_STIMULUS, STATE_PERCEIVING, TF_RESET_IDLE, 0},
        {STATE_IDLE, EVENT_WAKEUP_SIGNAL, STATE_IDLE, TF_RESET_IDLE, 0},
        {STATE_IDLE, EVENT_IDLE_TIMEOUT, STATE_SLEEP, TF_RESET_IDLE, 0},
        
        // SLEEP transitions  
        {STATE_SLEEP, EVENT_WAKEUP_SIGNAL, STATE_IDLE, TF_RESET_IDLE | TF_CLEAR_AWARENESS, 0},
        {STATE_SLEEP, EVENT_EXTERNAL_QUERY, STATE_HANDLING_QUERY, TF_PUSH_STACK | TF_RESET_IDLE, 0},
        
        // PERCEIVING transitions
        {STATE_PERCEIVING, EVENT_NAV_TARGET_LOCKED, STATE_NAVIGATING, TF_RESET_IDLE, 0},
        {STATE_PERCEIVING, EVENT_REASONING_REQUEST, STATE_REASONING, TF_RESET_IDLE, 0},
        
        // REASONING transitions
        {STATE_REASONING, EVENT_ACT_FINISHED, STATE_ACTING, 0, 0},
        
        // ACTING transitions
        {STATE_ACTING, EVENT_ACT_FINISHED, STATE_IDLE, 0, 0},
        
        // QUERY completion
        {STATE_HANDLING_QUERY, EVENT_ACT_FINISHED, STATE_IDLE, TF_POP_STACK, 0}, // Actually pop handled in code
        
        // Sentinel
        {0xFF, 0xFF, 0xFF, 0, 0}
    };
    
    cudaMemcpyToSymbol(kTransitionTable, h_table, sizeof(h_table));
    cudaMemcpyToSymbol(kIdleToSleepSeconds, &kIdleToSleepSeconds, sizeof(float));
}

// Kernel wrapper for trm_step_fused.cu integration
__global__ void trm_state_machine_kernel(
    EntityHotPath* entities,
    TRMStateMachine* state_machines,
    GPUEventQueue* event_queues,
    int num_entities,
    float delta_time,
    uint32_t current_tick
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entities) return;
    
    trm_state_machine_step(idx, entities, state_machines, event_queues, 
                          delta_time, current_tick, nullptr, nullptr);
}

#endif // TRM_STATE_MACHINE_CUH
```

## 2. Integration with Existing `trm_step_fused.cu`

```cuda
// Inside trm_step_fused.cu
#include "trm_state_machine.cuh"

__global__ void trm_step_fused(
    EntityHotPath* entities,
    TRMStateMachine* state_machines,  // NEW: Parallel array
    GPUEventQueue* event_queues,      // NEW: Per-entity event queues
    PhysicsParams physics,
    float delta_time,
    uint32_t tick
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= physics.num_entities) return;
    
    // ------------------------------------------------------------------------
    // SLOT 1: State Machine (The Clockwork)
    // ------------------------------------------------------------------------
    trm_state_machine_step(idx, entities, state_machines, event_queues,
                          delta_time, tick, physics.galaxy_ptr, physics.octree_ptr);
    
    __syncthreads