#pragma once

#include <math.h>
#include <stdint.h>

#define GPU_TASK_EMBED_DIMS 32
#define GPU_TASK_NUM_CHAINS 9
#define GPU_TASK_MAX_OPTIONS 7
#define GPU_TASK_INPUT_SLOT_BYTES 1280
#define GPU_TASK_OUTPUT_SLOT_BYTES 512
#define GPU_TASK_QUERY_EMBEDDING_OFFSET 0
#define GPU_TASK_TYPE_OFFSET 128
#define GPU_TASK_OPTION_COUNT_OFFSET 132
#define GPU_TASK_OPTION_EMBEDDINGS_OFFSET 136
#define GPU_TASK_THINKING_BUDGET_OFFSET 1040
#define GPU_TASK_ACTION_HISTORY_OFFSET 1044
#define GPU_TASK_ACTION_HISTORY_LEN_OFFSET 1051
#define GPU_TASK_TERNARY_SIGNAL_OFFSET 1052
#define GPU_TASK_GOAL_EMBEDDING_OFFSET 1056
#define GPU_TASK_DEFAULT_THINKING_BUDGET 10
#define GPU_TASK_MIN_THINKING_BUDGET 5
#define GPU_TASK_MAX_THINKING_BUDGET 20
#define GPU_TASK_ANSWER_INDEX_OUTPUT_OFFSET 0
#define GPU_TASK_CONFIDENCE_OUTPUT_OFFSET 4
#define GPU_TASK_CONVERGENCE_OUTPUT_OFFSET 8
#define GPU_TASK_ITERATIONS_OUTPUT_OFFSET 12
#define GPU_TASK_ANSWER_HASH_OUTPUT_OFFSET 16
#define GPU_TASK_GOAL_PROGRESS_OUTPUT_OFFSET 24

#define GALAXY_STAR_RECORD_BYTES 160
#define GALAXY_STAR_EMBEDDING_OFFSET 0
#define GALAXY_STAR_GALAXY_ID_OFFSET 128
#define GALAXY_STAR_TYPE_OFFSET 132
#define GALAXY_STAR_N_REFS_OFFSET 136
#define GALAXY_STAR_REFS_OFFSET 140
#define GALAXY_STAR_FLAGS_OFFSET 156
#define GALAXY_STAR_FLAG_ACTIVE 0x01
#define GALAXY_STAR_FLAG_LEARNABLE 0x02
#define GALAXY_NULL_REF 0xFFFFFFFFu

#define BRAIN_REASONING_OFFSET 0
#define BRAIN_CHAINS_OFFSET 128
#define BRAIN_PREV_FRAME_OFFSET 1280
#define BRAIN_ACTION_RING_OFFSET 1408
#define BRAIN_ACTION_RING_LEN_OFFSET 1415
#define BRAIN_TERNARY_OFFSET 1416
#define BRAIN_FRAME_COUNT_OFFSET 1420
#define BRAIN_SPECIALIST_TRACE_OFFSET 1424
#define BRAIN_TOTAL_BYTES 1460

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

__device__ __forceinline__ const float* galaxy_read_embedding(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return reinterpret_cast<const float*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_EMBEDDING_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_n_refs(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_N_REFS_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_ref(
    const unsigned char* galaxy_table,
    unsigned int star_index,
    unsigned int ref_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_REFS_OFFSET + (ref_index * 4u)
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

__device__ void galaxy_compose_embedding_device(
    float* output,
    const unsigned char* galaxy_table,
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

    const unsigned int n_refs = galaxy_read_n_refs(galaxy_table, star_index);
    if (n_refs == 0u) {
        return;
    }

    const unsigned int bounded_refs = n_refs > 4u ? 4u : n_refs;
    const float base_weight = 0.60f;
    const float ref_weight = 0.40f / static_cast<float>(bounded_refs);
    for (int index = 0; index < dim; ++index) {
        output[index] *= base_weight;
    }
    for (unsigned int ref_slot = 0u; ref_slot < bounded_refs; ++ref_slot) {
        const unsigned int ref_idx = galaxy_read_ref(galaxy_table, star_index, ref_slot);
        if (ref_idx == GALAXY_NULL_REF) {
            continue;
        }
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
