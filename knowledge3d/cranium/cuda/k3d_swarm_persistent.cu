#include <stdint.h>

#include "halting_gate.cu"
#include "lane_perf_ring.cu"
#include "n_selector.cu"
#if defined(K3D_REASONING_OPCODES_V1)
#include "reasoning_tick_entrypoints.cuh"
#endif

#define K3D_SWARM_WARPS_PER_BLOCK 4u
#define K3D_SWARM_N_HARD_MAX 1024u
#define K3D_SWARM_FLAG_RUN 0u
#define K3D_SWARM_FLAG_COMPLETE 1u
#define K3D_SWARM_FLAG_SHUTDOWN 2u

struct __align__(16) SwarmKernelControl {
    volatile uint32_t state;
    volatile uint32_t halting_counter;
    volatile uint32_t tick_epoch;
    volatile uint32_t halt_epoch;
};

static_assert(sizeof(SwarmKernelControl) == 16, "SwarmKernelControl ABI must remain stable");

static __device__ __forceinline__ uint32_t k3d_swarm_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __device__ __forceinline__ uint32_t k3d_swarm_atlas_u32(
    const uint8_t* __restrict__ galaxy_atlas,
    uint32_t index
) {
    if (galaxy_atlas == nullptr) {
        return 0u;
    }
    return reinterpret_cast<const uint32_t*>(galaxy_atlas)[index];
}

static __device__ __forceinline__ void k3d_swarm_dispatch_paradigm(
    uint32_t slot,
    const ReasoningTickIO& io,
    ReasoningLaneOutput* out
) {
#if defined(K3D_REASONING_OPCODES_V1)
    switch (slot) {
        case REASONING_SLOT_CBR:
            rpn_case_tick(io, out);
            break;
        case REASONING_SLOT_SUPERPOS:
            rpn_superpos_tick(io, out);
            break;
        case REASONING_SLOT_BIDUCE:
            rpn_frame_tick(io, out);
            break;
        case REASONING_SLOT_EBELIEF:
            rpn_ebelief_tick(io, out);
            break;
        case REASONING_SLOT_RETE:
            rpn_rete_tick(io, out);
            break;
        case REASONING_SLOT_TABLEAUX:
            rpn_tableaux_tick(io, out);
            break;
        case REASONING_SLOT_RESOLUTION:
            rpn_resolve_tick(io, out);
            break;
        case REASONING_SLOT_ALPCHAIN:
            rpn_alp_tick(io, out);
            break;
        case REASONING_SLOT_DPLL:
            rpn_dpll_tick(io, out);
            break;
        case REASONING_SLOT_CTX_SWITCH:
            rpn_ctx_switch_tick(io, out);
            break;
        case REASONING_SLOT_SUBSUME:
            rpn_subsume_tick(io, out);
            break;
        case REASONING_SLOT_UNIFY:
            rpn_unify_tick(io, out);
            break;
        case REASONING_SLOT_NONE:
        default:
            k3d_reasoning_zero_lane_output(out);
            k3d_reasoning_set_halt(io, out);
            break;
    }
#else
    (void)slot;
    (void)io;
    out->halt_flag = 1u;
    out->result_handle = 0u;
    out->belief_q15 = 0u;
    out->_pad0 = 0u;
    #pragma unroll
    for (uint32_t i = 0u; i < K3D_REASONING_LANE_OUTPUT_BYTES - 16u; ++i) {
        out->payload[i] = 0u;
    }
    reinterpret_cast<uint32_t*>(out->payload)[2] = 1u;
    reinterpret_cast<uint32_t*>(out->payload)[3] = 1u;
#endif
}

extern "C" __global__ void k3d_swarm_sovereign(
    const uint8_t* __restrict__ galaxy_atlas,
    SwarmKernelControl* __restrict__ control,
    const SwarmTickControl* __restrict__ tick_control,
    uint32_t* __restrict__ g_halting_counter,
    volatile uint32_t* __restrict__ d_n_active,
    ReasoningLaneOutput* __restrict__ lane_outputs,
    LanePerf* perf_ring,
    uint32_t* perf_ring_head,
    uint32_t perf_ring_mask,
    const SwarmPerfCalibration* __restrict__ perf_calibration,
    uint32_t* __restrict__ g_halting_value_q15
) {
    __shared__ uint32_t warp_halt_flags[K3D_SWARM_WARPS_PER_BLOCK];
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane_id = threadIdx.x & 31u;
    const uint32_t phys_lane_id = (blockIdx.x * K3D_SWARM_WARPS_PER_BLOCK) + warp_id;

    uint32_t tick_seed = 0u;
    while (true) {
        const uint32_t state = control->state;
        if (state == K3D_SWARM_FLAG_SHUTDOWN) {
            return;
        }
        if (state != K3D_SWARM_FLAG_RUN) {
            continue;
        }

        __shared__ uint32_t n_active_shared;
        if (threadIdx.x == 0u) {
            const uint32_t dynamic_n = swarm_select_dynamic_n(tick_control, perf_calibration);
            n_active_shared = swarm_clamp_u32(dynamic_n, 1u, K3D_SWARM_N_HARD_MAX);
            *d_n_active = n_active_shared;
        }
        __syncthreads();

        const uint32_t n_active = n_active_shared;
        const bool active = (phys_lane_id < n_active);
        const uint64_t start_clock = clock64();
        if (lane_id == 0u) {
            warp_halt_flags[warp_id] = active ? 0u : 1u;
        }
        __syncthreads();

        if (lane_id == 0u && lane_outputs != nullptr && phys_lane_id < K3D_SWARM_N_HARD_MAX) {
            ReasoningLaneOutput* out = lane_outputs + phys_lane_id;
            if (active) {
                const uint32_t paradigm_slot = swarm_assign_paradigm_slot(tick_control, phys_lane_id);
                const ReasoningTickIO io = {
                    galaxy_atlas,
                    phys_lane_id,
                    tick_seed,
                    paradigm_slot,
                    k3d_swarm_atlas_u32(galaxy_atlas, 0u),
                    k3d_swarm_atlas_u32(galaxy_atlas, 16u),
                    static_cast<int8_t>(k3d_swarm_atlas_u32(galaxy_atlas, 17u) & 0xFFu),
                    {0u, 0u, 0u},
                };
                k3d_swarm_dispatch_paradigm(paradigm_slot, io, out);
                warp_halt_flags[warp_id] = out->halt_flag;
            } else {
                out->halt_flag = 1u;
                out->result_handle = 0u;
                out->belief_q15 = 0u;
                out->_pad0 = 0u;
                #pragma unroll
                for (uint32_t i = 0u; i < K3D_REASONING_LANE_OUTPUT_BYTES - 16u; ++i) {
                    out->payload[i] = 0u;
                }
            }
        }
        __syncthreads();
        const uint32_t entropy_input = k3d_swarm_mix32(
            (tick_seed * 0x85EBCA6Bu) ^
            (phys_lane_id * 0x9E3779B9u) ^
            k3d_swarm_atlas_u32(galaxy_atlas, 0u) ^
            (active ? swarm_assign_paradigm_slot(tick_control, phys_lane_id) : 0u)
        );
        if (lane_id == 0u && perf_ring != nullptr && perf_ring_mask != 0u) {
            LanePerf sample;
            sample.n_active = n_active;
            sample.entropy_input = entropy_input;
            sample.belief_delta = (active && lane_outputs != nullptr)
                ? (static_cast<float>(lane_outputs[phys_lane_id].belief_q15) / 32768.0f)
                : 0.0f;
            sample.cycles_consumed = static_cast<uint32_t>(clock64() - start_clock);
            sample.specialist_id = static_cast<uint8_t>(phys_lane_id & 0xFFu);
            sample._pad[0] = 0u;
            sample._pad[1] = 0u;
            sample._pad[2] = 0u;
            lane_perf_write(perf_ring, perf_ring_head, perf_ring_mask, sample);
        }

        if (lane_id == 0u && phys_lane_id == 0u && lane_outputs != nullptr) {
            const uint32_t tick_marker = tick_seed + 1u;
            uint32_t halted_count = 0u;
            for (uint32_t lane = 0u; lane < n_active; ++lane) {
                volatile const uint32_t* payload_words =
                    reinterpret_cast<volatile const uint32_t*>(lane_outputs[lane].payload);
                while (payload_words[2] < tick_marker) {
                }
                if (lane_outputs[lane].halt_flag != 0u) {
                    ++halted_count;
                }
            }
            if (halted_count == n_active) {
                if (g_halting_counter != nullptr) {
                    *g_halting_counter = n_active;
                }
                if (g_halting_value_q15 != nullptr) {
                    // First-class halting scalar: max lane belief_q15
                    // across all active lanes at the moment of halt.
                    // Range [0, 32768] → consumer divides by 32768.0 to
                    // yield a float in [0.0, 1.0].
                    // See TEMP/CLAUDE_HALTING_READBACK_HOOK_SPEC_04.21.2026.md §3.1
                    uint32_t max_belief_q15 = 0u;
                    for (uint32_t lane = 0u; lane < n_active; ++lane) {
                        const uint32_t belief = lane_outputs[lane].belief_q15;
                        if (belief > max_belief_q15) {
                            max_belief_q15 = belief;
                        }
                    }
                    *g_halting_value_q15 = max_belief_q15;
                }
                control->state = K3D_SWARM_FLAG_COMPLETE;
                control->halt_epoch += 1u;
                __threadfence_system();
            }
        }
        ++tick_seed;
    }
}
