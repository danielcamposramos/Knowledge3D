#pragma once

#include <math.h>
#include <stdint.h>

#define GPU_TASK_EMBED_DIMS 64
#define GPU_TASK_TRM_DIMS 512
#define GPU_TASK_TRM_HIDDEN_DIMS 1024
#define GPU_TASK_TRM_WORKSPACE_FLOATS 4096
#define GPU_TASK_NUM_CHAINS 9
#define GPU_TASK_MAX_OPTIONS 7
#define GPU_TASK_INPUT_SLOT_BYTES 2688
#define GPU_TASK_OUTPUT_SLOT_BYTES 640
#define GPU_TASK_QUERY_EMBEDDING_OFFSET 0
#define GPU_TASK_TYPE_OFFSET 256
#define GPU_TASK_OPTION_COUNT_OFFSET 260
#define GPU_TASK_OPTION_EMBEDDINGS_OFFSET 264
#define GPU_TASK_OPTION_HASHES_OFFSET 2056
#define GPU_TASK_SUBJECT_ID_OFFSET 2112
#define GPU_TASK_DOMAIN_HINT_ID_OFFSET 2116
#define GPU_TASK_THINKING_BUDGET_OFFSET 2120
#define GPU_TASK_ACTION_HISTORY_OFFSET 2124
#define GPU_TASK_ACTION_HISTORY_LEN_OFFSET 2131
#define GPU_TASK_TERNARY_SIGNAL_OFFSET 2132
#define GPU_TASK_GOAL_EMBEDDING_OFFSET 2136
#define GPU_TASK_EXPECTED_HASH_OFFSET 2392
#define GPU_TASK_EXPECTED_INDEX_OFFSET 2400
#define GPU_TASK_DEFAULT_THINKING_BUDGET 10
#define GPU_TASK_MIN_THINKING_BUDGET 5
#define GPU_TASK_MAX_THINKING_BUDGET 20
#define GPU_TASK_ANSWER_INDEX_OUTPUT_OFFSET 0
#define GPU_TASK_CONFIDENCE_OUTPUT_OFFSET 4
#define GPU_TASK_CONVERGENCE_OUTPUT_OFFSET 8
#define GPU_TASK_ITERATIONS_OUTPUT_OFFSET 12
#define GPU_TASK_ANSWER_HASH_OUTPUT_OFFSET 16
#define GPU_TASK_GOAL_PROGRESS_OUTPUT_OFFSET 24
#define GPU_TASK_WINNER_STAR_INDEX_OUTPUT_OFFSET 28
#define GPU_TASK_WINNER_ROLE_ID_OUTPUT_OFFSET 32
#define GPU_TASK_ROUTE_DEPTH_OUTPUT_OFFSET 36
#define GPU_TASK_ANTI_PATTERN_SIGNAL_OUTPUT_OFFSET 40
#define GPU_TASK_ROUTER_STAR_INDEX_OUTPUT_OFFSET 44
#define GPU_TASK_EXECUTOR_STAR_INDEX_OUTPUT_OFFSET 48
#define GPU_TASK_VALIDATOR_STAR_INDEX_OUTPUT_OFFSET 52
#define GPU_TASK_ROUTE_BUDGET_USED_OUTPUT_OFFSET 56
#define GPU_TASK_ROUTE_BUDGET_MIN_OUTPUT_OFFSET 60
#define GPU_TASK_RECURSION_DEPTH_USED_OUTPUT_OFFSET 64
#define GPU_TASK_ROUTE_TRACE_STAR_INDICES_OUTPUT_OFFSET 68
#define GPU_TASK_ROUTE_TRACE_ROLE_IDS_OUTPUT_OFFSET 100

#define GPU_FAMILY_GAME_2D 1u
#define GPU_FAMILY_MATH 2u
#define GPU_FAMILY_QUESTION 3u
#define GPU_FAMILY_CHAT 4u
#define GPU_FAMILY_GENERAL 5u
#define GPU_FAMILY_GRAMMAR 6u
#define GPU_FAMILY_INTERACTION 7u

#define GALAXY_STAR_RECORD_BYTES 400
#define GALAXY_STAR_EMBEDDING_OFFSET 0
#define GALAXY_STAR_GALAXY_ID_OFFSET 256
#define GALAXY_STAR_TYPE_OFFSET 260
#define GALAXY_STAR_SELECTION_ROLE_OFFSET 264
#define GALAXY_STAR_LAYER_ID_OFFSET 268
#define GALAXY_STAR_FLAGS_OFFSET 272
#define GALAXY_STAR_ANSWER_ELIGIBLE_OFFSET 276
#define GALAXY_STAR_SEMANTIC_POLARITY_OFFSET 280
#define GALAXY_STAR_SEMANTIC_FOCUS_OFFSET 284
#define GALAXY_STAR_SEMANTIC_MASS_OFFSET 288
#define GALAXY_STAR_ATTRACTIVE_PRIOR_OFFSET 292
#define GALAXY_STAR_REPULSIVE_PRIOR_OFFSET 296
#define GALAXY_STAR_ROUTE_POLICY_OFFSET 300
#define GALAXY_STAR_HASH_OFFSET 304
#define GALAXY_STAR_ROUTER_REF_COUNT_OFFSET 312
#define GALAXY_STAR_ROUTER_REFS_OFFSET 316
#define GALAXY_STAR_EXECUTOR_REF_COUNT_OFFSET 324
#define GALAXY_STAR_EXECUTOR_REFS_OFFSET 328
#define GALAXY_STAR_VALIDATOR_REF_COUNT_OFFSET 336
#define GALAXY_STAR_VALIDATOR_REFS_OFFSET 340
#define GALAXY_STAR_ANTI_PATTERN_REF_COUNT_OFFSET 348
#define GALAXY_STAR_ANTI_PATTERN_REFS_OFFSET 352
#define GALAXY_STAR_POSITION_OFFSET 360
#define GALAXY_STAR_VELOCITY_OFFSET 372
#define GALAXY_STAR_META_RULE_ADDR_OFFSET 384
#define GALAXY_STAR_PROGRAM_FLAGS_OFFSET 388
#define GALAXY_STAR_PROGRAM_LENGTH_OFFSET 392
#define GALAXY_STAR_PROGRAM_OPCODE_COUNT_OFFSET 396
#define GALAXY_STAR_FLAG_ACTIVE 0x01
#define GALAXY_STAR_FLAG_LEARNABLE 0x02
#define GALAXY_STAR_ROUTE_FAMILY_SHIFT 8u
#define GALAXY_STAR_ROUTE_FAMILY_MASK (0xFFu << GALAXY_STAR_ROUTE_FAMILY_SHIFT)
#define GALAXY_NULL_REF 0xFFFFFFFFu

#define GALAXY_ROLE_UNKNOWN 0u
#define GALAXY_ROLE_ROUTER 1u
#define GALAXY_ROLE_EXECUTOR 2u
#define GALAXY_ROLE_VALIDATOR 3u
#define GALAXY_ROLE_ANSWER 4u
#define GALAXY_ROLE_ANTI_PATTERN 5u

#define GALAXY_ROLE_REF_LIMIT 2u

#define ROUTE_POLICY_DECOMPOSE_ON_FAIL 0x01u
#define ROUTE_POLICY_REQUIRES_EXECUTOR 0x02u
#define ROUTE_POLICY_REQUIRES_VALIDATOR 0x04u
#define ROUTE_POLICY_ANSWER_GATE 0x08u

#define GPU_ROUTE_MAX_DEPTH 8u
#define GPU_ROUTE_FRONTIER_WIDTH 32u
#define GPU_ROUTE_BRANCH_FANOUT 4u
#define GPU_ROUTE_TRACE_LIMIT 8u
#define GPU_ROUTE_VISITED_LIMIT 64u

#define LESSON_RECORD_BYTES 64
#define LESSON_FAMILY_ID_OFFSET 0
#define LESSON_ROUTER_INDEX_OFFSET 4
#define LESSON_EXECUTOR_INDEX_OFFSET 8
#define LESSON_VALIDATOR_INDEX_OFFSET 12
#define LESSON_WINNER_INDEX_OFFSET 16
#define LESSON_WINNER_ROLE_OFFSET 20
#define LESSON_EXPECTED_HASH_OFFSET 24
#define LESSON_PREDICTED_HASH_OFFSET 32
#define LESSON_REWARD_OFFSET 40
#define LESSON_ANTI_PATTERN_OFFSET 44
#define LESSON_ROUTE_DEPTH_OFFSET 48
#define LESSON_ROUTE_TRACE_HASH_OFFSET 56

#define LESSON_STATS_POSITIVE_STEPS_OFFSET 0
#define LESSON_STATS_NEGATIVE_STEPS_OFFSET 4
#define LESSON_STATS_ANTI_PATTERN_HITS_OFFSET 8
#define LESSON_STATS_LAST_POSITIVE_LOSS_OFFSET 12
#define LESSON_STATS_LAST_NEGATIVE_LOSS_OFFSET 16
#define LESSON_STATS_BYTES 32

#define BRAIN_REASONING_OFFSET 0
#define BRAIN_CHAINS_OFFSET 256
#define BRAIN_PREV_FRAME_OFFSET 2560
#define BRAIN_ACTION_RING_OFFSET 2816
#define BRAIN_ACTION_RING_LEN_OFFSET 2823
#define BRAIN_TERNARY_OFFSET 2824
#define BRAIN_FRAME_COUNT_OFFSET 2828
#define BRAIN_SPECIALIST_TRACE_OFFSET 2832
#define BRAIN_TRM_Q_OFFSET 2868
#define BRAIN_TRM_Y_OFFSET 4916
#define BRAIN_TRM_Z_OFFSET 6964
#define BRAIN_TOTAL_BYTES 9012

struct GalaxyRoleAdjacencyDeviceView {
    const unsigned int* router_offsets;
    const unsigned int* router_counts;
    const unsigned int* executor_offsets;
    const unsigned int* executor_counts;
    const unsigned int* validator_offsets;
    const unsigned int* validator_counts;
    const unsigned int* anti_pattern_offsets;
    const unsigned int* anti_pattern_counts;
    const unsigned int* ref_indices;
};

__device__ __forceinline__ float device_absf(float value) {
    return value < 0.0f ? -value : value;
}

__device__ __forceinline__ float device_clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

__device__ __forceinline__ float device_maxf(float a, float b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float device_minf(float a, float b) {
    return a < b ? a : b;
}

__device__ __forceinline__ bool device_isfinitef(float value) {
    return isfinite(value);
}

__device__ __forceinline__ float device_finite_or_default(float value, float default_value) {
    return device_isfinitef(value) ? value : default_value;
}

__device__ __forceinline__ float device_clamp_min(float value, float minimum) {
    return value < minimum ? minimum : value;
}

__device__ __forceinline__ float device_clamp_range(float value, float minimum, float maximum) {
    return device_minf(device_maxf(value, minimum), maximum);
}

__device__ __forceinline__ unsigned int device_clamp_u32(unsigned int value, unsigned int minimum, unsigned int maximum) {
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

__device__ __forceinline__ float pseudo_random_device(int chain_id, int dim) {
    unsigned int seed = static_cast<unsigned int>(chain_id * 73856093 ^ dim * 19349663);
    seed ^= seed >> 13;
    seed *= 1274126177u;
    seed ^= seed >> 16;
    return (static_cast<float>(seed & 0xFFFFu) / 65535.0f) - 0.5f;
}

__device__ __forceinline__ float dot32_device(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    for (int index = 0; index < dim; ++index) {
        dot += a[index] * b[index];
    }
    return dot;
}

__device__ __forceinline__ float norm32_device(const float* row, int dim) {
    float norm_sq = 0.0f;
    for (int index = 0; index < dim; ++index) {
        norm_sq += row[index] * row[index];
    }
    return sqrtf(norm_sq + 1.0e-12f);
}

__device__ __forceinline__ float cosine32_device(const float* a, const float* b, int dim) {
    const float denom = norm32_device(a, dim) * norm32_device(b, dim);
    if (denom <= 1.0e-12f) {
        return 0.0f;
    }
    return dot32_device(a, b, dim) / denom;
}

__device__ __forceinline__ void normalize_embedding64_device(
    const float* embedding64,
    float* normalized
) {
    double norm_sq = 0.0;
    #pragma unroll
    for (int d = 0; d < GPU_TASK_EMBED_DIMS; ++d) {
        const float value = device_finite_or_default(embedding64[d], 0.0f);
        normalized[d] = value;
        norm_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    const double norm = sqrt(norm_sq);
    const float inv_norm = norm > 1.0e-6 ? static_cast<float>(1.0 / norm) : 0.0f;
    #pragma unroll
    for (int d = 0; d < GPU_TASK_EMBED_DIMS; ++d) {
        normalized[d] *= inv_norm;
    }
}

__device__ __forceinline__ unsigned int pack_route_policy_device(
    bool decompose_on_fail,
    bool requires_executor,
    bool requires_validator,
    bool answer_gate,
    unsigned int branch_topk
) {
    unsigned int flags = 0u;
    if (decompose_on_fail) {
        flags |= ROUTE_POLICY_DECOMPOSE_ON_FAIL;
    }
    if (requires_executor) {
        flags |= ROUTE_POLICY_REQUIRES_EXECUTOR;
    }
    if (requires_validator) {
        flags |= ROUTE_POLICY_REQUIRES_VALIDATOR;
    }
    if (answer_gate) {
        flags |= ROUTE_POLICY_ANSWER_GATE;
    }
    return flags | (device_clamp_u32(branch_topk, 0u, 255u) << 8);
}

__device__ __forceinline__ const float* galaxy_read_embedding(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return reinterpret_cast<const float*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_EMBEDDING_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_selection_role(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_SELECTION_ROLE_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_answer_eligible(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_ANSWER_ELIGIBLE_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_flags(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_FLAGS_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_route_family_id(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return (galaxy_read_flags(galaxy_table, star_index) & GALAXY_STAR_ROUTE_FAMILY_MASK) >> GALAXY_STAR_ROUTE_FAMILY_SHIFT;
}

__device__ __forceinline__ int galaxy_read_polarity(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_SEMANTIC_POLARITY_OFFSET
    );
}

__device__ __forceinline__ float galaxy_read_mass(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const float*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_SEMANTIC_MASS_OFFSET
    );
}

__device__ __forceinline__ float galaxy_read_attractive_prior(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const float*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_ATTRACTIVE_PRIOR_OFFSET
    );
}

__device__ __forceinline__ float galaxy_read_repulsive_prior(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const float*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_REPULSIVE_PRIOR_OFFSET
    );
}

__device__ __forceinline__ unsigned long long galaxy_read_hash(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const unsigned long long*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_HASH_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_route_policy(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_ROUTE_POLICY_OFFSET
    );
}

__device__ __forceinline__ int galaxy_route_policy_flag(unsigned int route_policy_id, unsigned int mask) {
    return (route_policy_id & mask) != 0u;
}

__device__ __forceinline__ unsigned int galaxy_route_policy_branch_topk(unsigned int route_policy_id) {
    const unsigned int encoded = (route_policy_id >> 8) & 0xFFu;
    return encoded > 0u ? encoded : GPU_ROUTE_BRANCH_FANOUT;
}

__device__ __forceinline__ unsigned int route_budget_from_trit_device(int ternary_signal) {
    if (ternary_signal > 0) {
        return 5u;
    }
    if (ternary_signal < 0) {
        return 20u;
    }
    return 10u;
}

__device__ __forceinline__ unsigned int route_budget_min_from_trit_device(int ternary_signal) {
    if (ternary_signal < 0) {
        return 10u;
    }
    return 5u;
}

__device__ __forceinline__ const unsigned int* galaxy_role_offsets_ptr(
    const GalaxyRoleAdjacencyDeviceView& adjacency,
    unsigned int role_kind
) {
    if (role_kind == GALAXY_ROLE_EXECUTOR) {
        return adjacency.executor_offsets;
    }
    if (role_kind == GALAXY_ROLE_VALIDATOR) {
        return adjacency.validator_offsets;
    }
    if (role_kind == GALAXY_ROLE_ANTI_PATTERN) {
        return adjacency.anti_pattern_offsets;
    }
    return adjacency.router_offsets;
}

__device__ __forceinline__ const unsigned int* galaxy_role_counts_ptr(
    const GalaxyRoleAdjacencyDeviceView& adjacency,
    unsigned int role_kind
) {
    if (role_kind == GALAXY_ROLE_EXECUTOR) {
        return adjacency.executor_counts;
    }
    if (role_kind == GALAXY_ROLE_VALIDATOR) {
        return adjacency.validator_counts;
    }
    if (role_kind == GALAXY_ROLE_ANTI_PATTERN) {
        return adjacency.anti_pattern_counts;
    }
    return adjacency.router_counts;
}

__device__ __forceinline__ unsigned int galaxy_read_role_ref_count_csr(
    const GalaxyRoleAdjacencyDeviceView& adjacency,
    unsigned int star_index,
    unsigned int role_kind
) {
    const unsigned int* counts = galaxy_role_counts_ptr(adjacency, role_kind);
    return counts != nullptr ? counts[star_index] : 0u;
}

__device__ __forceinline__ unsigned int galaxy_read_role_ref_offset_csr(
    const GalaxyRoleAdjacencyDeviceView& adjacency,
    unsigned int star_index,
    unsigned int role_kind
) {
    const unsigned int* offsets = galaxy_role_offsets_ptr(adjacency, role_kind);
    return offsets != nullptr ? offsets[star_index] : 0u;
}

__device__ __forceinline__ unsigned int galaxy_read_role_ref_csr(
    const GalaxyRoleAdjacencyDeviceView& adjacency,
    unsigned int star_index,
    unsigned int role_kind,
    unsigned int ref_index
) {
    if (adjacency.ref_indices == nullptr) {
        return GALAXY_NULL_REF;
    }
    const unsigned int count = galaxy_read_role_ref_count_csr(adjacency, star_index, role_kind);
    if (ref_index >= count) {
        return GALAXY_NULL_REF;
    }
    const unsigned int offset = galaxy_read_role_ref_offset_csr(adjacency, star_index, role_kind);
    return adjacency.ref_indices[offset + ref_index];
}

__device__ __forceinline__ unsigned int galaxy_read_role_ref_count(
    const unsigned char* galaxy_table,
    unsigned int star_index,
    unsigned int role_kind
) {
    unsigned int offset = GALAXY_STAR_ROUTER_REF_COUNT_OFFSET;
    if (role_kind == GALAXY_ROLE_EXECUTOR) {
        offset = GALAXY_STAR_EXECUTOR_REF_COUNT_OFFSET;
    } else if (role_kind == GALAXY_ROLE_VALIDATOR) {
        offset = GALAXY_STAR_VALIDATOR_REF_COUNT_OFFSET;
    } else if (role_kind == GALAXY_ROLE_ANTI_PATTERN) {
        offset = GALAXY_STAR_ANTI_PATTERN_REF_COUNT_OFFSET;
    }
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + offset
    );
}

__device__ __forceinline__ unsigned int galaxy_read_role_ref(
    const unsigned char* galaxy_table,
    unsigned int star_index,
    unsigned int role_kind,
    unsigned int ref_index
) {
    unsigned int offset = GALAXY_STAR_ROUTER_REFS_OFFSET;
    if (role_kind == GALAXY_ROLE_EXECUTOR) {
        offset = GALAXY_STAR_EXECUTOR_REFS_OFFSET;
    } else if (role_kind == GALAXY_ROLE_VALIDATOR) {
        offset = GALAXY_STAR_VALIDATOR_REFS_OFFSET;
    } else if (role_kind == GALAXY_ROLE_ANTI_PATTERN) {
        offset = GALAXY_STAR_ANTI_PATTERN_REFS_OFFSET;
    }
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + offset + (ref_index * 4u)
    );
}

__device__ __forceinline__ float* galaxy_write_float_ptr(
    unsigned char* galaxy_table,
    unsigned int star_index,
    unsigned int offset
) {
    return reinterpret_cast<float*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + offset
    );
}

__device__ void galaxy_compose_embedding_device(
    float* output,
    const unsigned char* galaxy_table,
    const GalaxyRoleAdjacencyDeviceView& adjacency,
    unsigned int star_index,
    int dim
) {
    const unsigned int flags = galaxy_read_flags(galaxy_table, star_index);
    if ((flags & GALAXY_STAR_FLAG_ACTIVE) == 0u) {
        for (int index = 0; index < dim; ++index) {
            output[index] = 0.0f;
        }
        return;
    }

    const float* base = galaxy_read_embedding(galaxy_table, star_index);
    for (int index = 0; index < dim; ++index) {
        output[index] = base[index];
    }

    unsigned int refs[GPU_TASK_MAX_OPTIONS];
    unsigned int ref_count = 0u;
    for (unsigned int role_kind = GALAXY_ROLE_ROUTER; role_kind <= GALAXY_ROLE_ANTI_PATTERN; ++role_kind) {
        const unsigned int n_refs = galaxy_read_role_ref_count_csr(adjacency, star_index, role_kind);
        const unsigned int bounded_refs = n_refs > GPU_TASK_MAX_OPTIONS ? GPU_TASK_MAX_OPTIONS : n_refs;
        for (unsigned int ref_slot = 0u; ref_slot < bounded_refs; ++ref_slot) {
            const unsigned int ref_idx = galaxy_read_role_ref_csr(adjacency, star_index, role_kind, ref_slot);
            if (ref_idx == GALAXY_NULL_REF || ref_count >= GPU_TASK_MAX_OPTIONS) {
                continue;
            }
            bool duplicate = false;
            for (unsigned int existing = 0u; existing < ref_count; ++existing) {
                if (refs[existing] == ref_idx) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                refs[ref_count++] = ref_idx;
            }
        }
    }
    if (ref_count == 0u) {
        return;
    }

    const float base_weight = 0.60f;
    const float ref_weight = 0.40f / static_cast<float>(ref_count);
    for (int index = 0; index < dim; ++index) {
        output[index] *= base_weight;
    }
    for (unsigned int ref_slot = 0u; ref_slot < ref_count; ++ref_slot) {
        const unsigned int ref_idx = refs[ref_slot];
        const float* ref_embedding = galaxy_read_embedding(galaxy_table, ref_idx);
        for (int index = 0; index < dim; ++index) {
            output[index] += ref_weight * ref_embedding[index];
        }
    }
    float norm = 0.0f;
    for (int index = 0; index < dim; ++index) {
        norm += output[index] * output[index];
    }
    norm = sqrtf(norm + 1.0e-12f);
    if (norm > 1.0e-6f) {
        const float scale = 1.0f / norm;
        for (int index = 0; index < dim; ++index) {
            output[index] *= scale;
        }
    }
}

__device__ __forceinline__ int family_is_question_like(unsigned int family_id) {
    return family_id == GPU_FAMILY_QUESTION;
}

__device__ __forceinline__ int family_is_math_like(unsigned int family_id) {
    return family_id == GPU_FAMILY_MATH;
}

__device__ __forceinline__ int family_is_game_like(unsigned int family_id) {
    return family_id == GPU_FAMILY_GAME_2D;
}

__device__ __forceinline__ int8_t quantize_trit_device(float value) {
    if (value > 1e-6f) {
        return 1;
    }
    if (value < -1e-6f) {
        return -1;
    }
    return 0;
}

__device__ __forceinline__ int8_t clamp_trit_int_device(int value) {
    if (value > 0) {
        return 1;
    }
    if (value < 0) {
        return -1;
    }
    return 0;
}

__device__ __forceinline__ void copy_row_device(float* dst, const float* src, int dim) {
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        dst[index] = src[index];
    }
    __syncthreads();
}

__device__ __forceinline__ float goal_progress_device(
    const float* current_frame,
    const float* goal_embedding,
    const float* prev_frame,
    int dim
) {
    float current_dist = 0.0f;
    float prev_dist = 0.0f;
    for (int index = 0; index < dim; ++index) {
        const float current_delta = current_frame[index] - goal_embedding[index];
        const float prev_delta = prev_frame[index] - goal_embedding[index];
        current_dist += current_delta * current_delta;
        prev_dist += prev_delta * prev_delta;
    }
    current_dist = sqrtf(current_dist);
    prev_dist = sqrtf(prev_dist);
    if (current_dist < 1.0e-4f) {
        return 1.0f;
    }
    if (current_dist < prev_dist) {
        return 0.5f;
    }
    if (current_dist > prev_dist) {
        return -0.5f;
    }
    return 0.0f;
}

__device__ void blend_with_galaxy_device(
    float* embedding,
    const float* galaxy_knowledge,
    const float* context,
    const float* query,
    int dim,
    float self_weight,
    float galaxy_weight,
    float context_weight,
    float query_weight,
    unsigned int context_start_chain,
    unsigned int context_chain_count
) {
    if (threadIdx.x == 0) {
        const unsigned int bounded_count = context_chain_count == 0u ? 1u : context_chain_count;
        for (int index = 0; index < dim; ++index) {
            float context_mean = 0.0f;
            for (unsigned int offset = 0u; offset < context_chain_count; ++offset) {
                const unsigned int chain_index =
                    (context_start_chain + offset) % static_cast<unsigned int>(GPU_TASK_NUM_CHAINS);
                context_mean += context[(chain_index * dim) + index];
            }
            context_mean /= static_cast<float>(bounded_count);
            const float neighbor_mix =
                0.5f * (galaxy_knowledge[(index + dim - 1) % dim] + galaxy_knowledge[(index + 1) % dim]);
            embedding[index] = tanhf(
                (self_weight * embedding[index]) +
                (galaxy_weight * galaxy_knowledge[index]) +
                (context_weight * context_mean) +
                (query_weight * query[index]) +
                (0.05f * neighbor_mix)
            );
        }
    }
    __syncthreads();
}

__device__ void nine_chain_swarm_device(
    const float* query_embedding,
    float* chain_states,
    float* output_embedding,
    float* resonance_scores,
    int num_iterations
) {
    for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
        for (int dim = threadIdx.x; dim < GPU_TASK_EMBED_DIMS; dim += blockDim.x) {
            const float query_value = query_embedding[dim];
            if (chain == 0) {
                chain_states[chain * GPU_TASK_EMBED_DIMS + dim] = query_value;
            } else {
                const float noise = pseudo_random_device(chain, dim) * 0.05f;
                const float seeded = (0.90f * query_value) + noise;
                chain_states[chain * GPU_TASK_EMBED_DIMS + dim] = seeded;
            }
        }
    }
    __syncthreads();

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
            for (int dim = threadIdx.x; dim < GPU_TASK_EMBED_DIMS; dim += blockDim.x) {
                const int index = chain * GPU_TASK_EMBED_DIMS + dim;
                chain_states[index] = tanhf(chain_states[index]);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float consensus[GPU_TASK_EMBED_DIMS];
            for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
                float total = 0.0f;
                for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
                    total += chain_states[chain * GPU_TASK_EMBED_DIMS + dim];
                }
                consensus[dim] = total / static_cast<float>(GPU_TASK_NUM_CHAINS);
            }

            const float consensus_norm = norm32_device(consensus, GPU_TASK_EMBED_DIMS);
            float weight_sum = 0.0f;
            for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
                float* state = chain_states + (chain * GPU_TASK_EMBED_DIMS);
                const float state_norm = norm32_device(state, GPU_TASK_EMBED_DIMS);
                const float resonance = dot32_device(state, consensus, GPU_TASK_EMBED_DIMS) / (state_norm * consensus_norm + 1.0e-12f);
                resonance_scores[chain] = resonance;
                const float blend = (chain == 8) ? 0.18f : 0.12f;
                for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
                    state[dim] = ((1.0f - blend) * state[dim]) + (blend * consensus[dim]);
                }
                weight_sum += device_absf(resonance) + 1.0e-4f;
            }

            if (weight_sum <= 1.0e-12f) {
                weight_sum = 1.0f;
            }
            for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
                float total = 0.0f;
                for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
                    const float weight = (device_absf(resonance_scores[chain]) + 1.0e-4f) / weight_sum;
                    total += weight * chain_states[chain * GPU_TASK_EMBED_DIMS + dim];
                }
                output_embedding[dim] = total;
            }
        }
        __syncthreads();
    }
}

__device__ int halting_gate_device(
    const float* scores,
    int num_scores,
    float min_threshold,
    float gap_threshold,
    float agreement_threshold
) {
    if (num_scores <= 0) {
        return 0;
    }

    float top = scores[0];
    float second = -1.0e30f;
    for (int index = 1; index < num_scores; ++index) {
        const float score = scores[index];
        if (score >= top) {
            second = top;
            top = score;
        } else if (score > second) {
            second = score;
        }
    }

    int agreement = 0;
    const float tolerance = 0.15f;
    for (int index = 0; index < num_scores; ++index) {
        if ((top - scores[index]) <= tolerance) {
            agreement += 1;
        }
    }

    const float gap = top - second;
    const int minimum_flag = (min_threshold <= 0.0f || top >= min_threshold) ? 1 : 0;
    const int gap_flag = (gap_threshold <= 0.0f || gap >= gap_threshold) ? 1 : 0;
    const float required_agreement = agreement_threshold <= 0.0f
        ? 1.0f
        : ceilf(agreement_threshold * static_cast<float>(num_scores));
    const int agreement_flag = static_cast<float>(agreement) >= required_agreement ? 1 : 0;
    return (minimum_flag && gap_flag && agreement_flag) ? 1 : 0;
}

__device__ void defeasible_resolve_device(
    const float* conclusions,
    const int8_t* rule_strengths,
    float* verdicts,
    int num_candidates,
    int num_workers
) {
    if (threadIdx.x != 0) {
        __syncthreads();
        return;
    }

    for (int candidate = 0; candidate < num_candidates; ++candidate) {
        int strict_product = 1;
        int has_strict = 0;
        int defeasible_sum = 0;
        int defeater_triggered = 0;
        float confidence_sum = 0.0f;
        int confidence_count = 0;

        for (int worker = 0; worker < num_workers; ++worker) {
            const float raw = conclusions[(worker * GPU_TASK_MAX_OPTIONS) + candidate];
            const int8_t support = quantize_trit_device(raw);
            const int8_t strength = quantize_trit_device(static_cast<float>(rule_strengths[worker]));
            if (support == 0) {
                continue;
            }
            if (strength > 0) {
                strict_product *= static_cast<int>(support);
                has_strict = 1;
            } else if (strength == 0) {
                defeasible_sum += static_cast<int>(support);
            } else {
                defeater_triggered = 1;
            }
        }

        const int8_t definite = has_strict ? clamp_trit_int_device(strict_product) : 0;
        int8_t defeasible = clamp_trit_int_device(static_cast<int>(definite) + clamp_trit_int_device(defeasible_sum));
        if (defeater_triggered && definite == 0) {
            defeasible = 0;
        }

        if (defeasible != 0) {
            for (int worker = 0; worker < num_workers; ++worker) {
                const float raw = conclusions[(worker * GPU_TASK_MAX_OPTIONS) + candidate];
                if (quantize_trit_device(raw) != defeasible) {
                    continue;
                }
                confidence_sum += device_absf(raw);
                confidence_count += 1;
            }
        }

        const float mean_confidence = confidence_count > 0
            ? (confidence_sum / static_cast<float>(confidence_count))
            : 0.0f;
        verdicts[candidate] = static_cast<float>(defeasible) * mean_confidence;
    }
    __syncthreads();
}

__device__ void arc_reason_device(float* embedding, const float* context, int dim) {
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float spatial_delta = context[(3 * dim) + index] - context[(4 * dim) + index];
        embedding[index] = tanhf((0.96f * embedding[index]) + (0.04f * device_absf(spatial_delta)));
    }
    __syncthreads();
}

__device__ void arc3_action_select_device(
    float* embedding,
    const float* context,
    const float* frame_data,
    int dim
) {
    if (threadIdx.x == 0) {
        const float cx = frame_data[10] - 0.5f;
        const float cy = frame_data[11] - 0.5f;
        const float sx = frame_data[12];
        const float sy = frame_data[13];
        const float occupancy = device_clamp01(frame_data[28]);
        const float spread_mag = sqrtf((sx * sx) + (sy * sy));
        const float centeredness = 1.0f - device_clamp01((device_absf(cx) + device_absf(cy)) * 1.25f);
        const float movement_need = device_clamp01((device_absf(cx) + device_absf(cy)) * 1.6f + (0.35f * spread_mag));
        const float interaction_readiness = device_clamp01(centeredness * occupancy * (0.45f + (0.55f * frame_data[31])));
        const float click_readiness = device_clamp01(interaction_readiness * (1.0f - spread_mag) * frame_data[29]);

        embedding[0] = tanhf((0.35f * embedding[0]) + (1.75f * cx));
        embedding[1] = tanhf((0.35f * embedding[1]) + (1.75f * cy));
        embedding[2] = tanhf((0.20f * embedding[2]) + (1.10f * movement_need) - (0.65f * interaction_readiness));
        embedding[3] = tanhf((0.20f * embedding[3]) + (0.95f * click_readiness) - (0.55f * movement_need));
        embedding[4] = tanhf((0.30f * embedding[4]) + (0.90f * movement_need) + (0.15f * spread_mag));
        embedding[5] = tanhf((0.30f * embedding[5]) + (1.25f * cx));
        embedding[6] = tanhf((0.30f * embedding[6]) + (1.25f * cy));
        embedding[7] = tanhf((0.40f * embedding[7]) + (0.80f * occupancy));

        for (int index = 8; index < dim; ++index) {
            const float spatial_delta = context[(3 * dim) + index] - context[(4 * dim) + index];
            embedding[index] = tanhf(
                (0.82f * embedding[index]) +
                (0.02f * device_absf(spatial_delta)) +
                (0.02f * frame_data[index])
            );
        }

        embedding[10] = tanhf((0.15f * embedding[10]) - (0.40f * movement_need));
        embedding[11] = tanhf((0.15f * embedding[11]) + (0.30f * interaction_readiness));
    }
    __syncthreads();
}

__device__ void arc3_frame_delta_device(
    float* embedding,
    const float* frame_delta,
    int dim
) {
    if (threadIdx.x == 0) {
        float delta_magnitude = 0.0f;
        for (int index = 0; index < dim; ++index) {
            delta_magnitude += frame_delta[index] * frame_delta[index];
        }
        delta_magnitude = sqrtf(delta_magnitude + 1.0e-12f);
        const float delta_signal = device_clamp01(delta_magnitude * 2.0f);

        for (int index = 0; index < dim; ++index) {
            const float explore = (delta_signal < 0.1f)
                ? (0.08f * pseudo_random_device(index, dim))
                : 0.0f;
            embedding[index] = tanhf(
                (0.92f * embedding[index]) +
                (0.06f * delta_signal * frame_delta[index]) +
                explore
            );
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float arc3_action_prior_device(
    unsigned int option_index,
    const float* frame_data,
    int ternary_signal
) {
    const float cx = frame_data[10] - 0.5f;
    const float cy = frame_data[11] - 0.5f;
    const float sx = frame_data[12];
    const float sy = frame_data[13];
    const float spread_mag = sqrtf((sx * sx) + (sy * sy));
    const float occupancy = device_clamp01(frame_data[28]);
    const float centeredness = 1.0f - device_clamp01((device_absf(cx) + device_absf(cy)) * 1.25f);
    const float movement_need = device_clamp01((device_absf(cx) + device_absf(cy)) * 1.6f + (0.35f * spread_mag));
    const float interaction_readiness = device_clamp01(centeredness * occupancy * (0.45f + (0.55f * frame_data[31])));
    const float click_readiness = device_clamp01(interaction_readiness * (1.0f - spread_mag) * frame_data[29]);
    const float undo_readiness = ternary_signal < 0 ? 0.85f : 0.0f;

    switch (option_index) {
        case 0u:
            return (0.55f * movement_need * device_maxf(0.0f, -cy)) + (0.15f * movement_need);
        case 1u:
            return (0.55f * movement_need * device_maxf(0.0f, cy)) + (0.15f * movement_need);
        case 2u:
            return (0.55f * movement_need * device_maxf(0.0f, -cx)) + (0.15f * movement_need);
        case 3u:
            return (0.55f * movement_need * device_maxf(0.0f, cx)) + (0.15f * movement_need);
        case 4u:
            return (0.35f * interaction_readiness) - (0.25f * movement_need);
        case 5u:
            return (0.45f * click_readiness) - (0.20f * movement_need);
        case 6u:
            return (0.70f * undo_readiness) - (ternary_signal < 0 ? 0.0f : 0.40f);
        default:
            return 0.0f;
    }
}

__device__ void geometry_route_device(float* embedding, const float* context, int dim) {
    if (threadIdx.x == 0) {
        const float route = cosine32_device(embedding, context, dim);
        for (int index = 0; index < dim; ++index) {
            embedding[index] = tanhf(embedding[index] + (0.03f * route * context[index]));
        }
    }
    __syncthreads();
}

__device__ void fractal_emit_device(float* embedding, const float* context, int dim) {
    (void)context;
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float coarse = embedding[index / 2];
        const float fine = embedding[(index * 2) % dim];
        embedding[index] = tanhf((0.94f * embedding[index]) + (0.03f * coarse) + (0.03f * fine));
    }
    __syncthreads();
}

__device__ void atomic_fission_fusion_device(float* embedding, const float* context, int dim) {
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float atom_mean =
            (context[index] + context[dim + index] + context[(2 * dim) + index]) / 3.0f;
        embedding[index] = tanhf((0.92f * embedding[index]) + (0.08f * atom_mean));
    }
    __syncthreads();
}

__device__ void temporal_reason_device(float* embedding, const float* context, int dim) {
    (void)context;
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const int prev = (index + dim - 1) % dim;
        const float delta = embedding[index] - embedding[prev];
        embedding[index] = tanhf(embedding[index] + (0.05f * delta));
    }
    __syncthreads();
}

__device__ void graph_crystallize_device(float* embedding, const float* context, int dim) {
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        float neighbor_mean = 0.0f;
        for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
            neighbor_mean += context[(chain * dim) + index];
        }
        neighbor_mean /= static_cast<float>(GPU_TASK_NUM_CHAINS);
        embedding[index] = tanhf((0.90f * embedding[index]) + (0.10f * neighbor_mean));
    }
    __syncthreads();
}

__device__ void resonance_field_device(float* embedding, const float* context, int dim) {
    (void)context;
    if (threadIdx.x == 0) {
        float energy = 0.0f;
        for (int index = 0; index < dim; ++index) {
            energy += embedding[index] * embedding[index];
        }
        const float boost = 1.0f + (0.04f * device_clamp01(energy / static_cast<float>(dim)));
        for (int index = 0; index < dim; ++index) {
            embedding[index] *= boost;
        }
    }
    __syncthreads();
}

__device__ void vector_resonate_device(float* embedding, const float* context, int dim) {
    if (threadIdx.x == 0) {
        const float* creative = context + (7 * dim);
        const float* synthesis = context + (8 * dim);
        const float creative_score = cosine32_device(embedding, creative, dim);
        const float synthesis_score = cosine32_device(embedding, synthesis, dim);
        const float total = expf(creative_score) + expf(synthesis_score) + 1.0e-6f;
        const float creative_weight = expf(creative_score) / total;
        const float synthesis_weight = expf(synthesis_score) / total;
        for (int index = 0; index < dim; ++index) {
            const float blended = (creative_weight * creative[index]) + (synthesis_weight * synthesis[index]);
            embedding[index] = tanhf((0.75f * embedding[index]) + (0.25f * blended));
        }
    }
    __syncthreads();
}

__device__ void cognitive_executive_device(
    const float* resonance_scores,
    const float* chain_states,
    float* embedding,
    int dim
) {
    if (threadIdx.x == 0) {
        float logits[GPU_TASK_NUM_CHAINS];
        float trust[GPU_TASK_NUM_CHAINS];
        float max_logit = -1.0e30f;
        float denom = 0.0f;

        for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
            const float chain_norm = norm32_device(chain_states + (chain * dim), dim);
            logits[chain] = resonance_scores[chain] * (1.0f + logf(chain_norm + 1.0f));
            if (logits[chain] > max_logit) {
                max_logit = logits[chain];
            }
        }
        for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
            trust[chain] = expf(logits[chain] - max_logit);
            denom += trust[chain];
        }
        if (denom <= 1.0e-12f) {
            denom = 1.0f;
        }
        for (int index = 0; index < dim; ++index) {
            float blended = 0.0f;
            for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
                blended += (trust[chain] / denom) * chain_states[(chain * dim) + index];
            }
            embedding[index] = tanhf((0.80f * embedding[index]) + (0.20f * blended));
        }
    }
    __syncthreads();
}
