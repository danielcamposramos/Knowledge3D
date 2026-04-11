#include "trm_game_loop.cuh"

extern "C" __global__ void gpu_event_queue_reset(
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (head_ptr != nullptr) {
        *head_ptr = 0u;
    }
    if (tail_ptr != nullptr) {
        *tail_ptr = 0u;
    }
    if (ring_buffer != nullptr) {
        for (uint32_t index = 0u; index < TRM_EVENT_RING_CAPACITY; ++index) {
            ring_buffer[index].entity_id = 0u;
            ring_buffer[index].event_type = TRM_EVENT_NONE;
            ring_buffer[index].priority = 0u;
            ring_buffer[index].pad = 0u;
            ring_buffer[index].payload = 0u;
        }
    }
}

extern "C" __global__ void gpu_event_queue_enqueue_stress(
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    uint32_t entity_id,
    uint32_t total_events,
    uint64_t payload_base,
    uint32_t* push_results
) {
    const uint32_t producer_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t producer_count = gridDim.x * blockDim.x;
    for (uint32_t event_index = producer_id; event_index < total_events; event_index += producer_count) {
        GPUEvent event = {};
        event.entity_id = entity_id;
        event.event_type = static_cast<uint8_t>(TRM_EVENT_INTERNAL);
        event.priority = static_cast<uint8_t>(producer_id & 0xFFu);
        event.payload = payload_base + static_cast<uint64_t>(event_index);
        const bool pushed = trm_event_queue_push(ring_buffer, head_ptr, tail_ptr, event);
        if (push_results != nullptr) {
            push_results[event_index] = pushed ? 1u : 0u;
        }
    }
}

extern "C" __global__ void gpu_event_queue_enqueue_host_batch(
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    const GPUEvent* host_batch,
    uint32_t batch_size,
    uint32_t* push_results
) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= batch_size) {
        return;
    }

    const bool pushed = trm_event_queue_push(
        ring_buffer,
        head_ptr,
        tail_ptr,
        host_batch[tid]
    );
    if (push_results != nullptr) {
        push_results[tid] = pushed ? 1u : 0u;
    }
}

extern "C" __global__ void gpu_event_queue_dequeue_all(
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    GPUEvent* output_events,
    uint32_t max_events,
    uint32_t* output_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    uint32_t count = 0u;
    GPUEvent event = {};
    while (count < max_events && trm_event_queue_pop(ring_buffer, head_ptr, tail_ptr, &event)) {
        if (output_events != nullptr) {
            output_events[count] = event;
        }
        count += 1u;
    }

    if (output_count != nullptr) {
        *output_count = count;
    }
}
