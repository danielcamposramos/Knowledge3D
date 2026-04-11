#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "../kernels/entity_hot_path.h"

#define TRM_EVENT_RING_CAPACITY 4096u
#define TRM_EVENT_RING_MASK (TRM_EVENT_RING_CAPACITY - 1u)

#define TRM_STATE_FLAG_ATOMIC_SECTION 0x01u

#define TRM_DEFERRED_QUERY_POP 0x01u
#define TRM_DEFERRED_FOREIGN_EVENT 0x02u

#define TRM_MAX_EVENTS_PER_TICK 8u

#define TRM_TF_NONE 0x00u
#define TRM_TF_PUSH_STACK 0x01u
#define TRM_TF_POP_STACK 0x02u
#define TRM_TF_RESET_IDLE 0x04u

enum TRMEventType : uint8_t {
    TRM_EVENT_PERCEPTION_STIMULUS = 0,
    TRM_EVENT_COLLISION = 1,
    TRM_EVENT_INTERACTION = 2,
    TRM_EVENT_TIMER = 3,
    TRM_EVENT_IO = 4,
    TRM_EVENT_INTERNAL = 5,
    TRM_EVENT_WAKEUP = 6,
    TRM_EVENT_COUNT = 7,
    TRM_EVENT_NONE = 0xFFu,
};

enum TRMState : uint8_t {
    TRM_STATE_SLEEP = 0,
    TRM_STATE_IDLE = 1,
    TRM_STATE_PERCEIVING = 2,
    TRM_STATE_NAVIGATING = 3,
    TRM_STATE_REASONING = 4,
    TRM_STATE_ACTING = 5,
    TRM_STATE_HANDLING_QUERY = 6,
    TRM_STATE_COUNT = 7,
};

struct alignas(16) GPUEvent {
    uint32_t entity_id;
    uint8_t event_type;
    uint8_t priority;
    uint16_t pad;
    uint64_t payload;
};

static_assert(sizeof(GPUEvent) == 16, "GPUEvent layout mismatch");

struct StateTransition {
    uint8_t from_state;
    uint8_t event_type;
    uint8_t to_state;
    uint8_t action_flags;
};

#pragma pack(push, 1)
struct TRMStateMachine {
    uint8_t state_stack[4];
    uint8_t stack_depth;
    uint8_t current_state;
    uint8_t state_flags;
    uint8_t reserved;
    float idle_accumulator;
    uint64_t state_entry_tick;
    uint32_t deferred_event_mask;
    uint32_t interrupt_priority_level;
    uint32_t last_tick;
};
#pragma pack(pop)

static_assert(sizeof(TRMStateMachine) == 32, "TRMStateMachine layout mismatch");

#define TRM_TRANSITION_TABLE_SIZE 13

#ifdef K3D_TRM_DEFINE_TRANSITION_TABLE
__device__ __constant__ StateTransition kTrmStateTransitions[TRM_TRANSITION_TABLE_SIZE] = {
    {TRM_STATE_SLEEP, TRM_EVENT_WAKEUP, TRM_STATE_IDLE, TRM_TF_RESET_IDLE},
    {TRM_STATE_IDLE, TRM_EVENT_PERCEPTION_STIMULUS, TRM_STATE_PERCEIVING, TRM_TF_RESET_IDLE},
    {TRM_STATE_IDLE, TRM_EVENT_COLLISION, TRM_STATE_ACTING, TRM_TF_RESET_IDLE},
    {TRM_STATE_IDLE, TRM_EVENT_INTERACTION, TRM_STATE_ACTING, TRM_TF_RESET_IDLE},
    {TRM_STATE_IDLE, TRM_EVENT_TIMER, TRM_STATE_SLEEP, TRM_TF_RESET_IDLE},
    {TRM_STATE_IDLE, TRM_EVENT_INTERNAL, TRM_STATE_REASONING, TRM_TF_RESET_IDLE},
    {TRM_STATE_REASONING, TRM_EVENT_PERCEPTION_STIMULUS, TRM_STATE_PERCEIVING, TRM_TF_RESET_IDLE},
    {TRM_STATE_REASONING, TRM_EVENT_INTERACTION, TRM_STATE_ACTING, TRM_TF_RESET_IDLE},
    {TRM_STATE_PERCEIVING, TRM_EVENT_INTERNAL, TRM_STATE_NAVIGATING, TRM_TF_NONE},
    {TRM_STATE_NAVIGATING, TRM_EVENT_TIMER, TRM_STATE_ACTING, TRM_TF_NONE},
    {TRM_STATE_ACTING, TRM_EVENT_INTERNAL, TRM_STATE_IDLE, TRM_TF_NONE},
    {TRM_STATE_PERCEIVING, TRM_EVENT_COLLISION, TRM_STATE_ACTING, TRM_TF_NONE},
    {0xFFu, 0xFFu, 0xFFu, 0x00u},
};
#else
extern __device__ __constant__ StateTransition kTrmStateTransitions[TRM_TRANSITION_TABLE_SIZE];
#endif

__device__ __forceinline__ uint32_t trm_ring_next(uint32_t index) {
    return (index + 1u) & TRM_EVENT_RING_MASK;
}

__device__ __forceinline__ bool trm_event_queue_push(
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    const uint32_t* tail_ptr,
    const GPUEvent& event
) {
    if (ring_buffer == nullptr || head_ptr == nullptr || tail_ptr == nullptr) {
        return false;
    }

    uint32_t old_head = 0u;
    uint32_t new_head = 0u;
    do {
        old_head = *head_ptr;
        new_head = trm_ring_next(old_head);
        if (new_head == *tail_ptr) {
            return false;
        }
    } while (atomicCAS(head_ptr, old_head, new_head) != old_head);

    GPUEvent* slot = &ring_buffer[old_head];
    slot->entity_id = event.entity_id;
    slot->priority = event.priority;
    slot->pad = event.pad;
    slot->payload = event.payload;
    __threadfence();
    slot->event_type = event.event_type;
    __threadfence();
    return true;
}

__device__ __forceinline__ bool trm_event_queue_pop(
    GPUEvent* ring_buffer,
    const uint32_t* head_ptr,
    uint32_t* tail_ptr,
    GPUEvent* out_event
) {
    if (ring_buffer == nullptr || head_ptr == nullptr || tail_ptr == nullptr || out_event == nullptr) {
        return false;
    }

    uint32_t old_tail = 0u;
    uint32_t new_tail = 0u;
    do {
        old_tail = *tail_ptr;
        if (old_tail == *head_ptr) {
            return false;
        }
        if (ring_buffer[old_tail].event_type == TRM_EVENT_NONE) {
            return false;
        }
        out_event->entity_id = ring_buffer[old_tail].entity_id;
        out_event->event_type = ring_buffer[old_tail].event_type;
        out_event->priority = ring_buffer[old_tail].priority;
        out_event->pad = ring_buffer[old_tail].pad;
        out_event->payload = ring_buffer[old_tail].payload;
        new_tail = trm_ring_next(old_tail);
    } while (atomicCAS(tail_ptr, old_tail, new_tail) != old_tail);

    ring_buffer[old_tail].event_type = TRM_EVENT_NONE;
    return true;
}

__device__ __forceinline__ void trm_state_set(
    TRMStateMachine& state_machine,
    uint8_t new_state,
    uint64_t tick
) {
    state_machine.current_state = new_state;
    state_machine.state_entry_tick = tick;
    state_machine.idle_accumulator = 0.0f;
}

__device__ __forceinline__ void trm_state_push(
    TRMStateMachine& state_machine,
    uint8_t new_state,
    uint64_t tick
) {
    if (state_machine.stack_depth < 4u) {
        state_machine.state_stack[state_machine.stack_depth] = state_machine.current_state;
        state_machine.stack_depth += 1u;
    }
    trm_state_set(state_machine, new_state, tick);
}

__device__ __forceinline__ void trm_state_pop(
    TRMStateMachine& state_machine,
    uint64_t tick
) {
    if (state_machine.stack_depth > 0u) {
        state_machine.stack_depth -= 1u;
        trm_state_set(state_machine, state_machine.state_stack[state_machine.stack_depth], tick);
    } else {
        trm_state_set(state_machine, TRM_STATE_IDLE, tick);
    }
}

__device__ __forceinline__ const StateTransition* trm_lookup_transition(
    uint8_t current_state,
    uint8_t event_type
) {
    #pragma unroll
    for (int index = 0; index < TRM_TRANSITION_TABLE_SIZE; ++index) {
        const StateTransition* transition = &kTrmStateTransitions[index];
        if (transition->from_state == 0xFFu) {
            break;
        }
        if (transition->from_state == current_state && transition->event_type == event_type) {
            return transition;
        }
    }
    return nullptr;
}

__device__ __forceinline__ void trm_apply_transition(
    TRMStateMachine& state_machine,
    const StateTransition& transition,
    uint64_t tick
) {
    if ((transition.action_flags & TRM_TF_PUSH_STACK) != 0u) {
        trm_state_push(state_machine, transition.to_state, tick);
    } else if ((transition.action_flags & TRM_TF_POP_STACK) != 0u) {
        trm_state_pop(state_machine, tick);
    } else {
        trm_state_set(state_machine, transition.to_state, tick);
    }

    if ((transition.action_flags & TRM_TF_RESET_IDLE) != 0u) {
        state_machine.idle_accumulator = 0.0f;
    }
}

__device__ __forceinline__ void trm_bootstrap_state_machine(
    TRMStateMachine& state_machine,
    EntityHotPath& entity
) {
    if (state_machine.current_state >= TRM_STATE_COUNT) {
        const uint8_t fallback_state = entity.sleep_state < TRM_STATE_COUNT
            ? entity.sleep_state
            : TRM_STATE_IDLE;
        state_machine.current_state = fallback_state;
    }
    entity.sleep_state = state_machine.current_state;
}

__device__ __forceinline__ void trm_state_machine_step_device(
    EntityHotPath* entity_ptr,
    TRMStateMachine* state_machine_ptr,
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    float delta_time,
    uint64_t tick
) {
    if (entity_ptr == nullptr || state_machine_ptr == nullptr) {
        return;
    }

    EntityHotPath& entity = *entity_ptr;
    TRMStateMachine& state_machine = *state_machine_ptr;
    const uint32_t owner_entity_id = static_cast<uint32_t>(state_machine.reserved);
    trm_bootstrap_state_machine(state_machine, entity);
    state_machine.last_tick = static_cast<uint32_t>(tick & 0xFFFFFFFFu);

    GPUEvent event = {};
    bool handled_event = false;
    uint32_t drained = 0u;
    while (
        drained < TRM_MAX_EVENTS_PER_TICK
        && trm_event_queue_pop(ring_buffer, head_ptr, tail_ptr, &event)
    ) {
        drained += 1u;
        if (event.entity_id != owner_entity_id) {
            state_machine.deferred_event_mask |= TRM_DEFERRED_FOREIGN_EVENT;
            continue;
        }

        handled_event = true;
        state_machine.idle_accumulator = 0.0f;

        if (
            event.event_type == TRM_EVENT_IO
            && (state_machine.state_flags & TRM_STATE_FLAG_ATOMIC_SECTION) == 0u
        ) {
            state_machine.deferred_event_mask |= TRM_DEFERRED_QUERY_POP;
            if (state_machine.current_state != TRM_STATE_HANDLING_QUERY) {
                trm_state_push(state_machine, TRM_STATE_HANDLING_QUERY, tick);
            }
            state_machine.interrupt_priority_level = static_cast<uint32_t>(event.priority);
            entity.awareness = 1.0f;
            entity.sleep_state = state_machine.current_state;
            return;
        }

        const StateTransition* transition = trm_lookup_transition(
            state_machine.current_state,
            event.event_type
        );
        if (transition != nullptr) {
            trm_apply_transition(state_machine, *transition, tick);
        }
    }

    if (handled_event) {
        entity.sleep_state = state_machine.current_state;
        return;
    }

    if (
        state_machine.current_state == TRM_STATE_HANDLING_QUERY
        && (state_machine.deferred_event_mask & TRM_DEFERRED_QUERY_POP) != 0u
    ) {
        state_machine.deferred_event_mask &= ~TRM_DEFERRED_QUERY_POP;
        state_machine.interrupt_priority_level = 0u;
        trm_state_pop(state_machine, tick);
        entity.sleep_state = state_machine.current_state;
        return;
    }

    if (state_machine.current_state == TRM_STATE_IDLE) {
        state_machine.idle_accumulator += delta_time;
        if (state_machine.idle_accumulator >= 30.0f) {
            trm_state_set(state_machine, TRM_STATE_SLEEP, tick);
        }
    }

    entity.sleep_state = state_machine.current_state;
}
