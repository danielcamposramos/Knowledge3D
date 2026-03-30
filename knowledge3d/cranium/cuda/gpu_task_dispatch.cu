#include "device_functions.cuh"

extern "C" __global__ void gpu_task_dispatch(
    const unsigned char* __restrict__ input_buffer,
    unsigned char* __restrict__ output_buffer,
    unsigned int task_count,
    unsigned char* __restrict__ brain_state,
    const unsigned char* __restrict__ galaxy_table,
    unsigned int galaxy_star_count
) {
    const unsigned int task_id = blockIdx.x;
    if (task_id >= task_count) {
        return;
    }

    const unsigned int input_base = task_id * GPU_TASK_INPUT_SLOT_BYTES;
    const unsigned int output_base = task_id * GPU_TASK_OUTPUT_SLOT_BYTES;

    const float* query_embedding =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_QUERY_EMBEDDING_OFFSET);
    const unsigned int task_type =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_TYPE_OFFSET);
    const unsigned int option_count =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_OPTION_COUNT_OFFSET);
    const float* option_embeddings =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_OPTION_EMBEDDINGS_OFFSET);
    const float* goal_embedding =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_GOAL_EMBEDDING_OFFSET);

    const unsigned int raw_budget =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_THINKING_BUDGET_OFFSET);
    const unsigned int thinking_budget =
        (raw_budget >= GPU_TASK_MIN_THINKING_BUDGET && raw_budget <= GPU_TASK_MAX_THINKING_BUDGET)
            ? raw_budget
            : GPU_TASK_DEFAULT_THINKING_BUDGET;
    const unsigned char* action_history = input_buffer + input_base + GPU_TASK_ACTION_HISTORY_OFFSET;
    const unsigned int action_history_len =
        static_cast<unsigned int>(input_buffer[input_base + GPU_TASK_ACTION_HISTORY_LEN_OFFSET]);
    const int ternary_signal =
        static_cast<int>(*reinterpret_cast<const signed char*>(input_buffer + input_base + GPU_TASK_TERNARY_SIGNAL_OFFSET));
    const bool has_brain = brain_state != nullptr;
    float* brain_reasoning =
        has_brain ? reinterpret_cast<float*>(brain_state + BRAIN_REASONING_OFFSET) : nullptr;
    float* brain_chains =
        has_brain ? reinterpret_cast<float*>(brain_state + BRAIN_CHAINS_OFFSET) : nullptr;
    float* brain_prev_frame =
        has_brain ? reinterpret_cast<float*>(brain_state + BRAIN_PREV_FRAME_OFFSET) : nullptr;
    unsigned char* brain_action_ring =
        has_brain ? (brain_state + BRAIN_ACTION_RING_OFFSET) : nullptr;
    float* brain_specialist_trace =
        has_brain ? reinterpret_cast<float*>(brain_state + BRAIN_SPECIALIST_TRACE_OFFSET) : nullptr;

    __shared__ float chain_states[GPU_TASK_NUM_CHAINS * GPU_TASK_EMBED_DIMS];
    __shared__ float reasoning_state[GPU_TASK_EMBED_DIMS];
    __shared__ float swarm_output[GPU_TASK_EMBED_DIMS];
    __shared__ float frame_delta[GPU_TASK_EMBED_DIMS];
    __shared__ float resonance_scores[GPU_TASK_NUM_CHAINS];
    __shared__ float candidate_scores[GPU_TASK_MAX_OPTIONS];
    __shared__ unsigned int galaxy_nearest[8];
    __shared__ float galaxy_nearest_scores[8];
    __shared__ float galaxy_knowledge[GPU_TASK_EMBED_DIMS];
    __shared__ unsigned int scan_indices[128 * 4];
    __shared__ float scan_scores[128 * 4];
    __shared__ unsigned int best_index;
    __shared__ float best_score;
    __shared__ float goal_progress;
    __shared__ int converged;
    __shared__ unsigned int bounded_options;
    __shared__ unsigned int iterations_used;
    __shared__ unsigned int frame_count;
    __shared__ unsigned int active_action_history_len;
    __shared__ int active_ternary_signal;

    if (threadIdx.x == 0) {
        frame_count = has_brain
            ? *reinterpret_cast<const unsigned int*>(brain_state + BRAIN_FRAME_COUNT_OFFSET)
            : 0u;
        bounded_options = option_count > GPU_TASK_MAX_OPTIONS ? GPU_TASK_MAX_OPTIONS : option_count;
        best_index = 0u;
        best_score = 0.0f;
        goal_progress = 0.0f;
        converged = 0;
        iterations_used = 0u;
        active_action_history_len = (action_history_len > 7u) ? 7u : action_history_len;
        active_ternary_signal = ternary_signal;
        if (has_brain && frame_count > 0u) {
            const int stored_ternary =
                static_cast<int>(*reinterpret_cast<const signed char*>(brain_state + BRAIN_TERNARY_OFFSET));
            if (stored_ternary != 0) {
                active_ternary_signal = stored_ternary;
            }
            active_action_history_len =
                static_cast<unsigned int>(brain_state[BRAIN_ACTION_RING_LEN_OFFSET]);
            if (active_action_history_len > 7u) {
                active_action_history_len = 7u;
            }
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < GPU_TASK_EMBED_DIMS; index += blockDim.x) {
        if (has_brain && frame_count > 0u) {
            reasoning_state[index] = tanhf(
                (0.70f * brain_reasoning[index]) +
                (0.30f * query_embedding[index])
            );
            frame_delta[index] = query_embedding[index] - brain_prev_frame[index];
        } else {
            reasoning_state[index] = query_embedding[index];
            frame_delta[index] = 0.0f;
        }
        swarm_output[index] = reasoning_state[index];
    }
    __syncthreads();

    for (unsigned int think_step = 0u; think_step < thinking_budget; ++think_step) {
        const int swarm_rounds = (think_step == 0u) ? 3 : 2;
        nine_chain_swarm_device(reasoning_state, chain_states, swarm_output, resonance_scores, swarm_rounds);
        if (has_brain && frame_count > 0u) {
            for (int index = threadIdx.x; index < GPU_TASK_NUM_CHAINS * GPU_TASK_EMBED_DIMS; index += blockDim.x) {
                chain_states[index] = tanhf((0.70f * brain_chains[index]) + (0.30f * chain_states[index]));
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                for (unsigned int chain = 0u; chain < GPU_TASK_NUM_CHAINS; ++chain) {
                    resonance_scores[chain] = cosine32_device(
                        chain_states + (chain * GPU_TASK_EMBED_DIMS),
                        reasoning_state,
                        GPU_TASK_EMBED_DIMS
                    );
                }
            }
            __syncthreads();
        }

        const unsigned int scan_thread_count = blockDim.x < 128u ? blockDim.x : 128u;
        unsigned int local_indices[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        float local_scores[4] = {-1.0e30f, -1.0e30f, -1.0e30f, -1.0e30f};

        if (threadIdx.x < scan_thread_count && galaxy_table != nullptr && galaxy_star_count > 0u) {
            for (unsigned int star_idx = threadIdx.x; star_idx < galaxy_star_count; star_idx += scan_thread_count) {
                float star_embedding[GPU_TASK_EMBED_DIMS];
                galaxy_compose_embedding_device(star_embedding, galaxy_table, star_idx, GPU_TASK_EMBED_DIMS);
                const float sim = cosine32_device(reasoning_state, star_embedding, GPU_TASK_EMBED_DIMS);
                int worst_slot = 0;
                for (int slot = 1; slot < 4; ++slot) {
                    if (local_scores[slot] < local_scores[worst_slot]) {
                        worst_slot = slot;
                    }
                }
                if (sim > local_scores[worst_slot]) {
                    local_indices[worst_slot] = star_idx;
                    local_scores[worst_slot] = sim;
                }
            }
        }

        if (threadIdx.x < 128u) {
            const unsigned int scan_base = threadIdx.x * 4u;
            if (threadIdx.x < scan_thread_count) {
                for (int slot = 0; slot < 4; ++slot) {
                    scan_indices[scan_base + static_cast<unsigned int>(slot)] = local_indices[slot];
                    scan_scores[scan_base + static_cast<unsigned int>(slot)] = local_scores[slot];
                }
            } else {
                for (int slot = 0; slot < 4; ++slot) {
                    scan_indices[scan_base + static_cast<unsigned int>(slot)] = 0xFFFFFFFFu;
                    scan_scores[scan_base + static_cast<unsigned int>(slot)] = -1.0e30f;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int slot = 0; slot < 8; ++slot) {
                galaxy_nearest[slot] = 0xFFFFFFFFu;
                galaxy_nearest_scores[slot] = -1.0e30f;
            }
            for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                galaxy_knowledge[index] = 0.0f;
            }

            if (galaxy_table != nullptr && galaxy_star_count > 0u) {
                const unsigned int total_candidates = scan_thread_count * 4u;
                for (unsigned int candidate = 0u; candidate < total_candidates; ++candidate) {
                    const unsigned int candidate_index = scan_indices[candidate];
                    if (candidate_index == 0xFFFFFFFFu) {
                        continue;
                    }
                    const float candidate_score = scan_scores[candidate];
                    int worst_slot = 0;
                    for (int slot = 1; slot < 8; ++slot) {
                        if (galaxy_nearest_scores[slot] < galaxy_nearest_scores[worst_slot]) {
                            worst_slot = slot;
                        }
                    }
                    if (candidate_score > galaxy_nearest_scores[worst_slot]) {
                        bool duplicate = false;
                        for (int slot = 0; slot < 8; ++slot) {
                            if (galaxy_nearest[slot] == candidate_index) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) {
                            galaxy_nearest[worst_slot] = candidate_index;
                            galaxy_nearest_scores[worst_slot] = candidate_score;
                        }
                    }
                }

                float total_weight = 0.0f;
                for (int slot = 0; slot < 8; ++slot) {
                    if (galaxy_nearest[slot] == 0xFFFFFFFFu) {
                        continue;
                    }
                    const float weight = device_maxf(0.0f, galaxy_nearest_scores[slot]);
                    if (weight <= 1.0e-8f) {
                        continue;
                    }
                    total_weight += weight;
                    float star_embedding[GPU_TASK_EMBED_DIMS];
                    galaxy_compose_embedding_device(
                        star_embedding,
                        galaxy_table,
                        galaxy_nearest[slot],
                        GPU_TASK_EMBED_DIMS
                    );
                    for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                        galaxy_knowledge[index] += weight * star_embedding[index];
                    }
                }
                if (total_weight > 1.0e-6f) {
                    const float inv_total = 1.0f / total_weight;
                    for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                        galaxy_knowledge[index] *= inv_total;
                    }
                }
            }
        }
        __syncthreads();

        switch (task_type) {
            case 0u:
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.60f,
                    0.30f,
                    0.10f,
                    0.0f,
                    3u,
                    2u
                );
                geometry_route_device(swarm_output, galaxy_knowledge, GPU_TASK_EMBED_DIMS);
                fractal_emit_device(swarm_output, galaxy_knowledge, GPU_TASK_EMBED_DIMS);
                break;
            case 1u:
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.50f,
                    0.40f,
                    0.10f,
                    0.0f,
                    0u,
                    3u
                );
                atomic_fission_fusion_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                geometry_route_device(swarm_output, galaxy_knowledge, GPU_TASK_EMBED_DIMS);
                break;
            case 2u:
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.48f,
                    0.40f,
                    0.08f,
                    0.04f,
                    0u,
                    4u
                );
                atomic_fission_fusion_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                temporal_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 3u:
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.45f,
                    0.40f,
                    0.15f,
                    0.0f,
                    0u,
                    9u
                );
                graph_crystallize_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 4u:
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.50f,
                    0.35f,
                    0.10f,
                    0.05f,
                    7u,
                    2u
                );
                resonance_field_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                vector_resonate_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 8u:
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.45f,
                    0.40f,
                    0.10f,
                    0.05f,
                    3u,
                    2u
                );
                arc3_action_select_device(swarm_output, chain_states, query_embedding, GPU_TASK_EMBED_DIMS);
                arc3_frame_delta_device(swarm_output, frame_delta, GPU_TASK_EMBED_DIMS);
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.55f,
                    0.30f,
                    0.05f,
                    0.10f,
                    3u,
                    2u
                );
                break;
            default:
                blend_with_galaxy_device(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query_embedding,
                    GPU_TASK_EMBED_DIMS,
                    0.55f,
                    0.30f,
                    0.10f,
                    0.05f,
                    0u,
                    9u
                );
                break;
        }

        cognitive_executive_device(resonance_scores, chain_states, swarm_output, GPU_TASK_EMBED_DIMS);

        if (threadIdx.x == 0) {
            iterations_used = think_step + 1u;
            if (bounded_options == 0u) {
                converged = 0;
                best_index = 0u;
                best_score = 0.0f;
            } else {
                for (unsigned int option_index = 0u; option_index < bounded_options; ++option_index) {
                    float composed_option[GPU_TASK_EMBED_DIMS];
                    if (galaxy_table != nullptr && task_type == 8u && option_index < galaxy_star_count) {
                        galaxy_compose_embedding_device(
                            composed_option,
                            galaxy_table,
                            option_index,
                            GPU_TASK_EMBED_DIMS
                        );
                    } else {
                        const float* slot_option = option_embeddings + (option_index * GPU_TASK_EMBED_DIMS);
                        for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                            composed_option[index] = slot_option[index];
                        }
                    }
                    float score = cosine32_device(swarm_output, composed_option, GPU_TASK_EMBED_DIMS);
                    if (task_type == 8u) {
                        score += arc3_action_prior_device(option_index, query_embedding, active_ternary_signal);
                    }
                    const unsigned char* active_action_history =
                        (has_brain && frame_count > 0u) ? brain_action_ring : action_history;
                    for (unsigned int history_index = 0u; history_index < active_action_history_len; ++history_index) {
                        if (static_cast<unsigned int>(active_action_history[history_index]) == option_index) {
                            const float recency = 0.25f + (0.12f * static_cast<float>(history_index));
                            score *= recency;
                            break;
                        }
                    }
                    candidate_scores[option_index] = score;
                }
                for (unsigned int option_index = bounded_options; option_index < GPU_TASK_MAX_OPTIONS; ++option_index) {
                    candidate_scores[option_index] = -1.0e30f;
                }

                best_index = 0u;
                best_score = candidate_scores[0];
                for (unsigned int option_index = 1u; option_index < bounded_options; ++option_index) {
                        if (candidate_scores[option_index] > best_score) {
                            best_score = candidate_scores[option_index];
                            best_index = option_index;
                        }
                    }

                const float min_threshold = active_ternary_signal < 0 ? 0.05f : 0.1f;
                const float gap_threshold = active_ternary_signal < 0 ? 0.10f : 0.15f;
                converged = halting_gate_device(
                    resonance_scores,
                    GPU_TASK_NUM_CHAINS,
                    min_threshold,
                    gap_threshold,
                    0.7f
                );
            }
        }
        __syncthreads();

        for (int index = threadIdx.x; index < GPU_TASK_EMBED_DIMS; index += blockDim.x) {
            reasoning_state[index] = swarm_output[index];
        }
        __syncthreads();

        if (converged && iterations_used >= GPU_TASK_MIN_THINKING_BUDGET) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        bool has_goal_embedding = false;
        if (task_type == 8u) {
            for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                if (device_absf(goal_embedding[index]) > 1.0e-8f) {
                    has_goal_embedding = true;
                    break;
                }
            }
        }
        goal_progress = has_goal_embedding
            ? goal_progress_device(reasoning_state, goal_embedding, query_embedding, GPU_TASK_EMBED_DIMS)
            : 0.0f;

        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ANSWER_INDEX_OUTPUT_OFFSET) = best_index;
        *reinterpret_cast<float*>(output_buffer + output_base + GPU_TASK_CONFIDENCE_OUTPUT_OFFSET) = best_score;
        *reinterpret_cast<signed char*>(output_buffer + output_base + GPU_TASK_CONVERGENCE_OUTPUT_OFFSET) =
            bounded_options > 0u ? static_cast<signed char>(converged ? 1 : 0) : static_cast<signed char>(0);
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ITERATIONS_OUTPUT_OFFSET) = iterations_used;
        const unsigned long long answer_hash =
            (static_cast<unsigned long long>(task_type) << 32) |
            static_cast<unsigned long long>(best_index);
        *reinterpret_cast<unsigned long long*>(output_buffer + output_base + GPU_TASK_ANSWER_HASH_OUTPUT_OFFSET) = answer_hash;
        *reinterpret_cast<float*>(output_buffer + output_base + GPU_TASK_GOAL_PROGRESS_OUTPUT_OFFSET) = goal_progress;
    }
    if (has_brain) {
        for (int index = threadIdx.x; index < GPU_TASK_EMBED_DIMS; index += blockDim.x) {
            brain_reasoning[index] = reasoning_state[index];
            brain_prev_frame[index] = query_embedding[index];
        }
        for (int index = threadIdx.x; index < GPU_TASK_NUM_CHAINS * GPU_TASK_EMBED_DIMS; index += blockDim.x) {
            brain_chains[index] = chain_states[index];
        }
        for (int index = threadIdx.x; index < GPU_TASK_NUM_CHAINS; index += blockDim.x) {
            brain_specialist_trace[index] = resonance_scores[index];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            unsigned int ring_len = static_cast<unsigned int>(brain_state[BRAIN_ACTION_RING_LEN_OFFSET]);
            if (ring_len > 7u) {
                ring_len = 7u;
            }
            if (ring_len < 7u) {
                brain_action_ring[ring_len] = static_cast<unsigned char>(best_index & 0xFFu);
                ring_len += 1u;
            } else {
                for (unsigned int ring_index = 0u; ring_index + 1u < 7u; ++ring_index) {
                    brain_action_ring[ring_index] = brain_action_ring[ring_index + 1u];
                }
                brain_action_ring[6] = static_cast<unsigned char>(best_index & 0xFFu);
            }
            brain_state[BRAIN_ACTION_RING_LEN_OFFSET] = static_cast<unsigned char>(ring_len);
            const int learned_ternary = (task_type == 8u)
                ? static_cast<int>(quantize_trit_device(goal_progress))
                : active_ternary_signal;
            *reinterpret_cast<signed char*>(brain_state + BRAIN_TERNARY_OFFSET) =
                static_cast<signed char>(learned_ternary);
            *reinterpret_cast<unsigned int*>(brain_state + BRAIN_FRAME_COUNT_OFFSET) = frame_count + 1u;
        }
    }
}
