#include "device_functions.cuh"
#include "trm_recursive_core.cuh"

static constexpr unsigned int GPU_DOMAIN_HINT_HASH_GRAMMAR = 1263759358u;

__device__ __forceinline__ unsigned int effective_task_family_device(
    unsigned int family_id,
    unsigned int domain_hint_id
) {
    if ((family_id == GPU_FAMILY_QUESTION || family_id == GPU_FAMILY_GENERAL) && domain_hint_id == GPU_DOMAIN_HINT_HASH_GRAMMAR) {
        return GPU_FAMILY_GRAMMAR;
    }
    return family_id;
}

__device__ __forceinline__ int family_candidate_allowed_device(
    unsigned int task_family_id,
    unsigned int star_family_id
) {
    if (star_family_id == 0u) {
        return 1;
    }
    if (task_family_id == star_family_id) {
        return 1;
    }
    if (task_family_id == GPU_FAMILY_QUESTION && star_family_id == GPU_FAMILY_GENERAL) {
        return 1;
    }
    if (task_family_id == GPU_FAMILY_GENERAL && star_family_id == GPU_FAMILY_QUESTION) {
        return 1;
    }
    return 0;
}

__device__ __forceinline__ float family_match_bonus_device(
    unsigned int task_family_id,
    unsigned int star_family_id,
    unsigned int role_id
) {
    if (star_family_id == 0u) {
        return 0.0f;
    }
    if (task_family_id == star_family_id) {
        if (role_id == GALAXY_ROLE_ROUTER) return 0.75f;
        if (role_id == GALAXY_ROLE_EXECUTOR) return 0.60f;
        if (role_id == GALAXY_ROLE_VALIDATOR || role_id == GALAXY_ROLE_ANSWER) return 0.55f;
        return 0.35f;
    }
    if (
        (task_family_id == GPU_FAMILY_QUESTION && star_family_id == GPU_FAMILY_GENERAL)
        || (task_family_id == GPU_FAMILY_GENERAL && star_family_id == GPU_FAMILY_QUESTION)
    ) {
        if (role_id == GALAXY_ROLE_ROUTER) return 0.28f;
        if (role_id == GALAXY_ROLE_EXECUTOR) return 0.24f;
        if (role_id == GALAXY_ROLE_VALIDATOR || role_id == GALAXY_ROLE_ANSWER) return 0.22f;
        return 0.12f;
    }
    if (
        star_family_id == GPU_FAMILY_GRAMMAR
        && (family_is_math_like(task_family_id) || family_is_question_like(task_family_id) || family_is_game_like(task_family_id))
    ) {
        return -1.10f;
    }
    if (family_is_game_like(task_family_id)) {
        return -0.95f;
    }
    if (family_is_math_like(task_family_id) || family_is_question_like(task_family_id)) {
        return -0.55f;
    }
    return -0.25f;
}

__device__ __forceinline__ float family_role_bias_device(unsigned int family_id, unsigned int role_id) {
    if (family_is_math_like(family_id)) {
        if (role_id == GALAXY_ROLE_ROUTER) return 0.18f;
        if (role_id == GALAXY_ROLE_EXECUTOR) return 0.26f;
        if (role_id == GALAXY_ROLE_VALIDATOR || role_id == GALAXY_ROLE_ANSWER) return 0.20f;
        if (role_id == GALAXY_ROLE_ANTI_PATTERN) return -0.40f;
    }
    if (family_is_question_like(family_id)) {
        if (role_id == GALAXY_ROLE_ROUTER) return 0.16f;
        if (role_id == GALAXY_ROLE_EXECUTOR) return 0.18f;
        if (role_id == GALAXY_ROLE_VALIDATOR || role_id == GALAXY_ROLE_ANSWER) return 0.24f;
        if (role_id == GALAXY_ROLE_ANTI_PATTERN) return -0.35f;
    }
    if (family_is_game_like(family_id)) {
        if (role_id == GALAXY_ROLE_ROUTER) return 0.12f;
        if (role_id == GALAXY_ROLE_EXECUTOR) return 0.14f;
        if (role_id == GALAXY_ROLE_VALIDATOR || role_id == GALAXY_ROLE_ANSWER) return 0.10f;
        if (role_id == GALAXY_ROLE_ANTI_PATTERN) return -0.30f;
    }
    if (role_id == GALAXY_ROLE_ANTI_PATTERN) return -0.25f;
    return 0.06f;
}

__device__ __forceinline__ float star_selection_score_device(
    const unsigned char* galaxy_table,
    const GalaxyRoleAdjacencyDeviceView& adjacency,
    unsigned int star_index,
    const float* reasoning_state,
    unsigned int family_id
) {
    float composed_embedding[GPU_TASK_EMBED_DIMS];
    galaxy_compose_embedding_device(composed_embedding, galaxy_table, adjacency, star_index, GPU_TASK_EMBED_DIMS);
    const float similarity = cosine32_device(reasoning_state, composed_embedding, GPU_TASK_EMBED_DIMS);
    const unsigned int role_id = galaxy_read_selection_role(galaxy_table, star_index);
    const unsigned int star_family_id = galaxy_read_route_family_id(galaxy_table, star_index);
    const float attractive = galaxy_read_attractive_prior(galaxy_table, star_index);
    const float repulsive = galaxy_read_repulsive_prior(galaxy_table, star_index);
    return similarity
        + family_role_bias_device(family_id, role_id)
        + family_match_bonus_device(family_id, star_family_id, role_id)
        + (0.20f * attractive)
        - (0.30f * repulsive);
}

__device__ __forceinline__ unsigned int choose_best_candidate_device(
    const unsigned char* galaxy_table,
    const unsigned int* nearest_indices,
    const float* nearest_scores,
    unsigned int nearest_count,
    unsigned int desired_role,
    unsigned int family_id,
    int require_answer_eligible
) {
    unsigned int best_index = 0xFFFFFFFFu;
    float best_score = -1.0e30f;
    for (unsigned int slot = 0u; slot < nearest_count; ++slot) {
        const unsigned int candidate_index = nearest_indices[slot];
        if (candidate_index == 0xFFFFFFFFu) {
            continue;
        }
        const unsigned int star_family_id = galaxy_read_route_family_id(galaxy_table, candidate_index);
        if (!family_candidate_allowed_device(family_id, star_family_id)) {
            continue;
        }
        const unsigned int role_id = galaxy_read_selection_role(galaxy_table, candidate_index);
        if (desired_role != GALAXY_ROLE_UNKNOWN && role_id != desired_role) {
            if (!(desired_role == GALAXY_ROLE_ROUTER && role_id == GALAXY_ROLE_ANSWER)) {
                continue;
            }
        }
        if (require_answer_eligible && galaxy_read_answer_eligible(galaxy_table, candidate_index) == 0u) {
            continue;
        }
        const float score =
            nearest_scores[slot]
            + family_role_bias_device(family_id, role_id)
            + family_match_bonus_device(family_id, star_family_id, role_id)
            + (0.20f * galaxy_read_attractive_prior(galaxy_table, candidate_index))
            - (0.20f * galaxy_read_repulsive_prior(galaxy_table, candidate_index));
        if (score > best_score) {
            best_score = score;
            best_index = candidate_index;
        }
    }
    return best_index;
}

__device__ __forceinline__ float route_transition_bonus_device(
    unsigned int family_id,
    unsigned int parent_role,
    unsigned int child_role
) {
    float bonus = 0.0f;
    if (parent_role == GALAXY_ROLE_ROUTER && child_role == GALAXY_ROLE_EXECUTOR) {
        bonus += family_is_math_like(family_id) ? 0.22f : 0.16f;
    } else if (parent_role == GALAXY_ROLE_EXECUTOR && (child_role == GALAXY_ROLE_VALIDATOR || child_role == GALAXY_ROLE_ANSWER)) {
        bonus += family_is_question_like(family_id) ? 0.22f : 0.18f;
    } else if (parent_role == GALAXY_ROLE_ROUTER && child_role == GALAXY_ROLE_VALIDATOR) {
        bonus += 0.12f;
    } else if (child_role == GALAXY_ROLE_ANTI_PATTERN) {
        bonus -= 0.35f;
    }
    return bonus;
}

__device__ __forceinline__ int role_is_emittable_device(
    unsigned int role_id,
    unsigned int answer_eligible
) {
    if (answer_eligible == 0u) {
        return 0;
    }
    return role_id == GALAXY_ROLE_EXECUTOR || role_id == GALAXY_ROLE_VALIDATOR || role_id == GALAXY_ROLE_ANSWER;
}

__device__ __forceinline__ int route_path_has_role(
    const unsigned int* role_trace,
    unsigned int trace_len,
    unsigned int role_id
) {
    for (unsigned int slot = 0u; slot < trace_len && slot < GPU_ROUTE_TRACE_LIMIT; ++slot) {
        if (role_trace[slot] == role_id) {
            return 1;
        }
    }
    return 0;
}

__device__ __forceinline__ unsigned long long route_trace_hash_device(
    const unsigned int* star_trace,
    const unsigned int* role_trace,
    unsigned int trace_len
) {
    unsigned long long value = 14695981039346656037ull;
    const unsigned int bounded = trace_len > GPU_ROUTE_TRACE_LIMIT ? GPU_ROUTE_TRACE_LIMIT : trace_len;
    for (unsigned int slot = 0u; slot < bounded; ++slot) {
        value ^= static_cast<unsigned long long>(star_trace[slot] + 0x9e3779b9u);
        value *= 1099511628211ull;
        value ^= static_cast<unsigned long long>(role_trace[slot] + 0x85ebca6bu);
        value *= 1099511628211ull;
    }
    return value;
}

extern "C" __global__ void gpu_task_dispatch(
    const unsigned char* __restrict__ input_buffer,
    unsigned char* __restrict__ output_buffer,
    unsigned int task_count,
    unsigned char* __restrict__ brain_state,
    const unsigned char* __restrict__ galaxy_table,
    unsigned int galaxy_star_count,
    const unsigned int* __restrict__ router_offsets,
    const unsigned int* __restrict__ router_counts,
    const unsigned int* __restrict__ executor_offsets,
    const unsigned int* __restrict__ executor_counts,
    const unsigned int* __restrict__ validator_offsets,
    const unsigned int* __restrict__ validator_counts,
    const unsigned int* __restrict__ anti_pattern_offsets,
    const unsigned int* __restrict__ anti_pattern_counts,
    const unsigned int* __restrict__ ref_indices,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    unsigned char* __restrict__ lesson_buffer,
    unsigned int* __restrict__ lesson_counter,
    unsigned int lesson_capacity
) {
    const unsigned int task_id = blockIdx.x;
    if (task_id >= task_count) {
        return;
    }

    const unsigned int input_base = task_id * GPU_TASK_INPUT_SLOT_BYTES;
    const unsigned int output_base = task_id * GPU_TASK_OUTPUT_SLOT_BYTES;

    const float* query_embedding =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_QUERY_EMBEDDING_OFFSET);
    const unsigned int family_id =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_TYPE_OFFSET);
    const unsigned int domain_hint_id =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_DOMAIN_HINT_ID_OFFSET);
    const unsigned int option_count =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_OPTION_COUNT_OFFSET);
    const float* option_embeddings =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_OPTION_EMBEDDINGS_OFFSET);
    const unsigned long long* option_hashes =
        reinterpret_cast<const unsigned long long*>(input_buffer + input_base + GPU_TASK_OPTION_HASHES_OFFSET);
    const float* goal_embedding =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_GOAL_EMBEDDING_OFFSET);
    const unsigned long long expected_hash =
        *reinterpret_cast<const unsigned long long*>(input_buffer + input_base + GPU_TASK_EXPECTED_HASH_OFFSET);
    const int expected_index =
        *reinterpret_cast<const int*>(input_buffer + input_base + GPU_TASK_EXPECTED_INDEX_OFFSET);

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

    const unsigned int task_family_id = effective_task_family_device(family_id, domain_hint_id);
    const bool is_game2d_grid_task =
        (task_family_id == GPU_FAMILY_GAME_2D && option_count == 0u);
    const bool has_brain = brain_state != nullptr;
    const GalaxyRoleAdjacencyDeviceView adjacency = {
        router_offsets,
        router_counts,
        executor_offsets,
        executor_counts,
        validator_offsets,
        validator_counts,
        anti_pattern_offsets,
        anti_pattern_counts,
        ref_indices,
    };
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
    float* brain_trm_q =
        has_brain ? reinterpret_cast<float*>(brain_state + BRAIN_TRM_Q_OFFSET) : nullptr;
    float* brain_trm_y =
        has_brain ? reinterpret_cast<float*>(brain_state + BRAIN_TRM_Y_OFFSET) : nullptr;
    float* brain_trm_z =
        has_brain ? reinterpret_cast<float*>(brain_state + BRAIN_TRM_Z_OFFSET) : nullptr;

    __shared__ float chain_states[GPU_TASK_NUM_CHAINS * GPU_TASK_EMBED_DIMS];
    __shared__ float reasoning_state[GPU_TASK_EMBED_DIMS];
    __shared__ float swarm_output[GPU_TASK_EMBED_DIMS];
    __shared__ float frame_delta[GPU_TASK_EMBED_DIMS];
    __shared__ float resonance_scores[GPU_TASK_NUM_CHAINS];
    __shared__ float candidate_scores[GPU_TASK_MAX_OPTIONS];
    __shared__ unsigned int nearest_indices[8];
    __shared__ float nearest_scores[8];
    __shared__ unsigned int scan_indices[128 * 4];
    __shared__ float scan_scores[128 * 4];
    __shared__ float galaxy_knowledge[GPU_TASK_EMBED_DIMS];
    __shared__ float trm_q[GPU_TASK_TRM_DIMS];
    __shared__ float trm_y[GPU_TASK_TRM_DIMS];
    __shared__ float trm_z[GPU_TASK_TRM_DIMS];
    __shared__ float trm_workspace[GPU_TASK_TRM_WORKSPACE_FLOATS];
    __shared__ float trm_reasoning[GPU_TASK_EMBED_DIMS];
    __shared__ unsigned int bounded_options;
    __shared__ unsigned int iterations_used;
    __shared__ unsigned int best_index;
    __shared__ float best_score;
    __shared__ int converged;
    __shared__ float goal_progress;
    __shared__ unsigned int active_action_history_len;
    __shared__ int active_ternary_signal;
    __shared__ unsigned int frame_count;
    __shared__ unsigned int router_index;
    __shared__ unsigned int executor_index;
    __shared__ unsigned int validator_index;
    __shared__ unsigned int anti_index;
    __shared__ unsigned int winner_index;
    __shared__ unsigned int winner_role;
    __shared__ unsigned int route_depth;
    __shared__ unsigned int route_budget_used;
    __shared__ unsigned int route_budget_min;
    __shared__ unsigned int recursion_depth_used;
    __shared__ int anti_pattern_signal;
    __shared__ unsigned int route_trace_star_indices[GPU_ROUTE_TRACE_LIMIT];
    __shared__ unsigned int route_trace_role_ids[GPU_ROUTE_TRACE_LIMIT];
    __shared__ int trm_steps_counter_shared;
    __shared__ unsigned int trm_steps_used;
    __shared__ float trm_drift_value;
    __shared__ unsigned int trm_enabled;

    if (threadIdx.x == 0) {
        frame_count = has_brain
            ? *reinterpret_cast<const unsigned int*>(brain_state + BRAIN_FRAME_COUNT_OFFSET)
            : 0u;
        bounded_options = option_count > GPU_TASK_MAX_OPTIONS ? GPU_TASK_MAX_OPTIONS : option_count;
        iterations_used = 0u;
        best_index = 0u;
        best_score = 0.0f;
        converged = 0;
        goal_progress = 0.0f;
        active_action_history_len = (action_history_len > 7u) ? 7u : action_history_len;
        active_ternary_signal = ternary_signal;
        router_index = 0xFFFFFFFFu;
        executor_index = 0xFFFFFFFFu;
        validator_index = 0xFFFFFFFFu;
        anti_index = 0xFFFFFFFFu;
        winner_index = 0xFFFFFFFFu;
        winner_role = GALAXY_ROLE_UNKNOWN;
        route_depth = 0u;
        route_budget_used = 0u;
        route_budget_min = 0u;
        recursion_depth_used = 0u;
        anti_pattern_signal = 0;
        trm_steps_counter_shared = 0;
        trm_steps_used = 0u;
        trm_drift_value = 0.0f;
        trm_enabled = (W1 != nullptr && W2 != nullptr && W3 != nullptr && W4 != nullptr) ? 1u : 0u;
        for (unsigned int slot = 0u; slot < GPU_ROUTE_TRACE_LIMIT; ++slot) {
            route_trace_star_indices[slot] = 0xFFFFFFFFu;
            route_trace_role_ids[slot] = GALAXY_ROLE_UNKNOWN;
        }
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
            reasoning_state[index] = tanhf((0.70f * brain_reasoning[index]) + (0.30f * query_embedding[index]));
            frame_delta[index] = query_embedding[index] - brain_prev_frame[index];
        } else {
            reasoning_state[index] = query_embedding[index];
            frame_delta[index] = 0.0f;
        }
        swarm_output[index] = reasoning_state[index];
        galaxy_knowledge[index] = 0.0f;
    }
    for (int index = threadIdx.x; index < GPU_TASK_TRM_DIMS; index += blockDim.x) {
        trm_q[index] = 0.0f;
        trm_y[index] = 0.0f;
        trm_z[index] = 0.0f;
    }
    __syncthreads();

    for (unsigned int think_step = 0u; think_step < thinking_budget; ++think_step) {
        const int swarm_rounds = (think_step == 0u) ? 3 : 2;
        nine_chain_swarm_device(reasoning_state, chain_states, swarm_output, resonance_scores, swarm_rounds);

        const unsigned int scan_thread_count = blockDim.x < 128u ? blockDim.x : 128u;
        unsigned int local_indices[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        float local_scores[4] = {-1.0e30f, -1.0e30f, -1.0e30f, -1.0e30f};

        if (threadIdx.x < scan_thread_count && galaxy_table != nullptr && galaxy_star_count > 0u) {
            for (unsigned int star_idx = threadIdx.x; star_idx < galaxy_star_count; star_idx += scan_thread_count) {
                if ((galaxy_read_flags(galaxy_table, star_idx) & GALAXY_STAR_FLAG_ACTIVE) == 0u) {
                    continue;
                }
                const float score = star_selection_score_device(galaxy_table, adjacency, star_idx, reasoning_state, task_family_id);
                int worst_slot = 0;
                for (int slot = 1; slot < 4; ++slot) {
                    if (local_scores[slot] < local_scores[worst_slot]) {
                        worst_slot = slot;
                    }
                }
                if (score > local_scores[worst_slot]) {
                    local_indices[worst_slot] = star_idx;
                    local_scores[worst_slot] = score;
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
                nearest_indices[slot] = 0xFFFFFFFFu;
                nearest_scores[slot] = -1.0e30f;
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
                        if (nearest_scores[slot] < nearest_scores[worst_slot]) {
                            worst_slot = slot;
                        }
                    }
                    if (candidate_score > nearest_scores[worst_slot]) {
                        bool duplicate = false;
                        for (int slot = 0; slot < 8; ++slot) {
                            if (nearest_indices[slot] == candidate_index) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) {
                            nearest_indices[worst_slot] = candidate_index;
                            nearest_scores[worst_slot] = candidate_score;
                        }
                    }
                }
            }

            router_index = 0xFFFFFFFFu;
            executor_index = 0xFFFFFFFFu;
            validator_index = 0xFFFFFFFFu;
            anti_index = 0xFFFFFFFFu;
            winner_index = 0xFFFFFFFFu;
            winner_role = GALAXY_ROLE_UNKNOWN;
            route_depth = 0u;
            route_budget_used = 0u;
            route_budget_min = 0u;
            recursion_depth_used = 0u;
            anti_pattern_signal = 0;
            for (unsigned int slot = 0u; slot < GPU_ROUTE_TRACE_LIMIT; ++slot) {
                route_trace_star_indices[slot] = 0xFFFFFFFFu;
                route_trace_role_ids[slot] = GALAXY_ROLE_UNKNOWN;
            }

            const unsigned int arb_budget = route_budget_from_trit_device(active_ternary_signal);
            route_budget_min = route_budget_min_from_trit_device(active_ternary_signal);
            unsigned int effective_route_budget = thinking_budget > arb_budget ? thinking_budget : arb_budget;
            if (effective_route_budget < route_budget_min) {
                effective_route_budget = route_budget_min;
            }
            if (effective_route_budget > GPU_TASK_MAX_THINKING_BUDGET) {
                effective_route_budget = GPU_TASK_MAX_THINKING_BUDGET;
            }

            unsigned int queue_indices[GPU_ROUTE_FRONTIER_WIDTH];
            float queue_scores[GPU_ROUTE_FRONTIER_WIDTH];
            unsigned int queue_depths[GPU_ROUTE_FRONTIER_WIDTH];
            unsigned int queue_path_lengths[GPU_ROUTE_FRONTIER_WIDTH];
            unsigned int queue_path_indices[GPU_ROUTE_FRONTIER_WIDTH][GPU_ROUTE_TRACE_LIMIT];
            unsigned int queue_path_roles[GPU_ROUTE_FRONTIER_WIDTH][GPU_ROUTE_TRACE_LIMIT];
            unsigned long long visited_hashes[GPU_ROUTE_VISITED_LIMIT];
            unsigned int visited_count = 0u;
            unsigned int queue_head = 0u;
            unsigned int queue_tail = 0u;
            float best_router_score = -1.0e30f;
            float best_executor_score = -1.0e30f;
            float best_validator_score = -1.0e30f;
            float best_anti_score = -1.0e30f;
            float best_winner_score = -1.0e30f;
            unsigned int best_router_path_len = 0u;
            unsigned int best_executor_path_len = 0u;
            unsigned int best_validator_path_len = 0u;
            unsigned int best_winner_path_len = 0u;
            unsigned int best_router_path[GPU_ROUTE_TRACE_LIMIT];
            unsigned int best_router_roles[GPU_ROUTE_TRACE_LIMIT];
            unsigned int best_executor_path[GPU_ROUTE_TRACE_LIMIT];
            unsigned int best_executor_roles[GPU_ROUTE_TRACE_LIMIT];
            unsigned int best_validator_path[GPU_ROUTE_TRACE_LIMIT];
            unsigned int best_validator_roles[GPU_ROUTE_TRACE_LIMIT];
            unsigned int best_winner_path[GPU_ROUTE_TRACE_LIMIT];
            unsigned int best_winner_roles[GPU_ROUTE_TRACE_LIMIT];

            for (unsigned int slot = 0u; slot < GPU_ROUTE_TRACE_LIMIT; ++slot) {
                best_router_path[slot] = 0xFFFFFFFFu;
                best_router_roles[slot] = GALAXY_ROLE_UNKNOWN;
                best_executor_path[slot] = 0xFFFFFFFFu;
                best_executor_roles[slot] = GALAXY_ROLE_UNKNOWN;
                best_validator_path[slot] = 0xFFFFFFFFu;
                best_validator_roles[slot] = GALAXY_ROLE_UNKNOWN;
                best_winner_path[slot] = 0xFFFFFFFFu;
                best_winner_roles[slot] = GALAXY_ROLE_UNKNOWN;
            }

            for (unsigned int seed_round = 0u; seed_round < GPU_ROUTE_BRANCH_FANOUT && queue_tail < GPU_ROUTE_FRONTIER_WIDTH; ++seed_round) {
                unsigned int seed_index = 0xFFFFFFFFu;
                float seed_score = -1.0e30f;
                for (unsigned int slot = 0u; slot < 8u; ++slot) {
                    const unsigned int candidate_index = nearest_indices[slot];
                    if (candidate_index == 0xFFFFFFFFu) {
                        continue;
                    }
                    bool duplicate = false;
                    for (unsigned int existing = 0u; existing < queue_tail; ++existing) {
                        if (queue_indices[existing] == candidate_index) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) {
                        continue;
                    }
                    const unsigned int role_id = galaxy_read_selection_role(galaxy_table, candidate_index);
                    if (role_id != GALAXY_ROLE_ROUTER && role_id != GALAXY_ROLE_ANSWER) {
                        continue;
                    }
                    if (is_game2d_grid_task && role_id != GALAXY_ROLE_ROUTER) {
                        continue;
                    }
                    const unsigned int star_family_id = galaxy_read_route_family_id(galaxy_table, candidate_index);
                    if (!family_candidate_allowed_device(task_family_id, star_family_id)) {
                        continue;
                    }
                    const float score =
                        nearest_scores[slot]
                        + family_role_bias_device(task_family_id, role_id)
                        + family_match_bonus_device(task_family_id, star_family_id, role_id);
                    if (score > seed_score) {
                        seed_score = score;
                        seed_index = candidate_index;
                    }
                }
                if (seed_index == 0xFFFFFFFFu) {
                    break;
                }
                const unsigned int queue_slot = queue_tail++;
                const unsigned int role_id = galaxy_read_selection_role(galaxy_table, seed_index);
                queue_indices[queue_slot] = seed_index;
                queue_scores[queue_slot] = seed_score;
                queue_depths[queue_slot] = 1u;
                queue_path_lengths[queue_slot] = 1u;
                for (unsigned int path_slot = 0u; path_slot < GPU_ROUTE_TRACE_LIMIT; ++path_slot) {
                    queue_path_indices[queue_slot][path_slot] = 0xFFFFFFFFu;
                    queue_path_roles[queue_slot][path_slot] = GALAXY_ROLE_UNKNOWN;
                }
                queue_path_indices[queue_slot][0] = seed_index;
                queue_path_roles[queue_slot][0] = role_id;
                if (visited_count < GPU_ROUTE_VISITED_LIMIT) {
                    visited_hashes[visited_count++] = galaxy_read_hash(galaxy_table, seed_index);
                }
            }
            if (queue_tail == 0u) {
                const unsigned int fallback_seed = choose_best_candidate_device(
                    galaxy_table,
                    nearest_indices,
                    nearest_scores,
                    8u,
                    is_game2d_grid_task ? GALAXY_ROLE_ROUTER : GALAXY_ROLE_UNKNOWN,
                    task_family_id,
                    0
                );
                if (fallback_seed != 0xFFFFFFFFu) {
                    queue_indices[0] = fallback_seed;
                    queue_scores[0] = star_selection_score_device(galaxy_table, adjacency, fallback_seed, reasoning_state, task_family_id);
                    queue_depths[0] = 1u;
                    queue_path_lengths[0] = 1u;
                    for (unsigned int path_slot = 0u; path_slot < GPU_ROUTE_TRACE_LIMIT; ++path_slot) {
                        queue_path_indices[0][path_slot] = 0xFFFFFFFFu;
                        queue_path_roles[0][path_slot] = GALAXY_ROLE_UNKNOWN;
                    }
                    queue_path_indices[0][0] = fallback_seed;
                    queue_path_roles[0][0] = galaxy_read_selection_role(galaxy_table, fallback_seed);
                    queue_tail = 1u;
                    visited_hashes[visited_count++] = galaxy_read_hash(galaxy_table, fallback_seed);
                }
            }

            while (queue_head < queue_tail && route_budget_used < effective_route_budget) {
                const unsigned int node_slot = queue_head++;
                const unsigned int node_index = queue_indices[node_slot];
                if (node_index == 0xFFFFFFFFu) {
                    continue;
                }
                route_budget_used += 1u;
                const unsigned int node_depth = queue_depths[node_slot];
                if (node_depth > recursion_depth_used) {
                    recursion_depth_used = node_depth;
                }
                const unsigned int node_path_len = queue_path_lengths[node_slot];
                const unsigned int node_role = galaxy_read_selection_role(galaxy_table, node_index);
                const unsigned int node_policy = galaxy_read_route_policy(galaxy_table, node_index);
                const unsigned int node_answer_eligible = galaxy_read_answer_eligible(galaxy_table, node_index);
                const float node_score = queue_scores[node_slot];

                if (node_role == GALAXY_ROLE_ROUTER && node_score > best_router_score) {
                    best_router_score = node_score;
                    router_index = node_index;
                    best_router_path_len = node_path_len;
                    for (unsigned int copy_slot = 0u; copy_slot < GPU_ROUTE_TRACE_LIMIT; ++copy_slot) {
                        best_router_path[copy_slot] = queue_path_indices[node_slot][copy_slot];
                        best_router_roles[copy_slot] = queue_path_roles[node_slot][copy_slot];
                    }
                }
                if (
                    node_role == GALAXY_ROLE_EXECUTOR
                    && !(is_game2d_grid_task && galaxy_route_policy_flag(node_policy, ROUTE_POLICY_MATERIALIZE_ACTION))
                    && node_score > best_executor_score
                ) {
                    best_executor_score = node_score;
                    executor_index = node_index;
                    best_executor_path_len = node_path_len;
                    for (unsigned int copy_slot = 0u; copy_slot < GPU_ROUTE_TRACE_LIMIT; ++copy_slot) {
                        best_executor_path[copy_slot] = queue_path_indices[node_slot][copy_slot];
                        best_executor_roles[copy_slot] = queue_path_roles[node_slot][copy_slot];
                    }
                }
                if (
                    (node_role == GALAXY_ROLE_VALIDATOR || node_role == GALAXY_ROLE_ANSWER)
                    && !(is_game2d_grid_task && node_role == GALAXY_ROLE_ANSWER)
                    && !(is_game2d_grid_task && node_path_len < 4u)
                    && node_score > best_validator_score
                ) {
                    best_validator_score = node_score;
                    validator_index = node_index;
                    best_validator_path_len = node_path_len;
                    for (unsigned int copy_slot = 0u; copy_slot < GPU_ROUTE_TRACE_LIMIT; ++copy_slot) {
                        best_validator_path[copy_slot] = queue_path_indices[node_slot][copy_slot];
                        best_validator_roles[copy_slot] = queue_path_roles[node_slot][copy_slot];
                    }
                }
                if (node_role == GALAXY_ROLE_ANTI_PATTERN && node_score > best_anti_score) {
                    best_anti_score = node_score;
                    anti_index = node_index;
                    anti_pattern_signal = 1;
                }

                int eligible_winner = role_is_emittable_device(node_role, node_answer_eligible);
                if (eligible_winner) {
                    if (is_game2d_grid_task && node_role == GALAXY_ROLE_ANSWER) {
                        eligible_winner = 0;
                    }
                    if (is_game2d_grid_task && node_path_len < 4u) {
                        eligible_winner = 0;
                    }
                    const int has_executor = route_path_has_role(queue_path_roles[node_slot], node_path_len, GALAXY_ROLE_EXECUTOR);
                    const int has_validator = route_path_has_role(queue_path_roles[node_slot], node_path_len, GALAXY_ROLE_VALIDATOR);
                    if (galaxy_route_policy_flag(node_policy, ROUTE_POLICY_REQUIRES_EXECUTOR) && node_role != GALAXY_ROLE_EXECUTOR && !has_executor) {
                        eligible_winner = 0;
                    }
                    if (galaxy_route_policy_flag(node_policy, ROUTE_POLICY_REQUIRES_VALIDATOR) && node_role != GALAXY_ROLE_VALIDATOR && !has_validator) {
                        eligible_winner = 0;
                    }
                    if (galaxy_route_policy_flag(node_policy, ROUTE_POLICY_ANSWER_GATE) && route_budget_used < route_budget_min) {
                        eligible_winner = 0;
                    }
                }
                if (eligible_winner && node_score > best_winner_score) {
                    best_winner_score = node_score;
                    winner_index = node_index;
                    winner_role = node_role;
                    best_winner_path_len = node_path_len;
                    for (unsigned int copy_slot = 0u; copy_slot < GPU_ROUTE_TRACE_LIMIT; ++copy_slot) {
                        best_winner_path[copy_slot] = queue_path_indices[node_slot][copy_slot];
                        best_winner_roles[copy_slot] = queue_path_roles[node_slot][copy_slot];
                    }
                }

                if (node_depth >= GPU_ROUTE_MAX_DEPTH || queue_tail >= GPU_ROUTE_FRONTIER_WIDTH) {
                    continue;
                }

                const unsigned int branch_topk = galaxy_route_policy_branch_topk(node_policy) > GPU_ROUTE_BRANCH_FANOUT
                    ? GPU_ROUTE_BRANCH_FANOUT
                    : galaxy_route_policy_branch_topk(node_policy);
                unsigned int child_indices[GPU_ROUTE_BRANCH_FANOUT];
                float child_scores[GPU_ROUTE_BRANCH_FANOUT];
                for (unsigned int child_slot = 0u; child_slot < GPU_ROUTE_BRANCH_FANOUT; ++child_slot) {
                    child_indices[child_slot] = 0xFFFFFFFFu;
                    child_scores[child_slot] = -1.0e30f;
                }

                for (unsigned int role_kind = GALAXY_ROLE_ROUTER; role_kind <= GALAXY_ROLE_ANTI_PATTERN; ++role_kind) {
                    const unsigned int ref_count = galaxy_read_role_ref_count_csr(adjacency, node_index, role_kind);
                    for (unsigned int ref_slot = 0u; ref_slot < ref_count; ++ref_slot) {
                        const unsigned int child_index = galaxy_read_role_ref_csr(adjacency, node_index, role_kind, ref_slot);
                        if (child_index == GALAXY_NULL_REF || child_index >= galaxy_star_count) {
                            continue;
                        }
                        const unsigned long long child_hash = galaxy_read_hash(galaxy_table, child_index);
                        bool duplicate = false;
                        for (unsigned int visited_slot = 0u; visited_slot < visited_count; ++visited_slot) {
                            if (visited_hashes[visited_slot] == child_hash) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (duplicate) {
                            continue;
                        }
                        const unsigned int child_role = galaxy_read_selection_role(galaxy_table, child_index);
                        const unsigned int child_family_id = galaxy_read_route_family_id(galaxy_table, child_index);
                        const unsigned int child_policy = galaxy_read_route_policy(galaxy_table, child_index);
                        if (child_role != GALAXY_ROLE_ANTI_PATTERN && !family_candidate_allowed_device(task_family_id, child_family_id)) {
                            continue;
                        }
                        float child_score =
                            star_selection_score_device(galaxy_table, adjacency, child_index, reasoning_state, task_family_id)
                            + (0.12f * node_score)
                            + route_transition_bonus_device(task_family_id, node_role, child_role);
                        if (is_game2d_grid_task) {
                            if (galaxy_route_policy_flag(child_policy, ROUTE_POLICY_MATERIALIZE_GRID)) {
                                child_score += 0.35f;
                            }
                            if (galaxy_route_policy_flag(child_policy, ROUTE_POLICY_MATERIALIZE_ACTION) || child_role == GALAXY_ROLE_ANSWER) {
                                child_score -= 0.45f;
                            }
                        }
                        int worst_slot = 0;
                        for (int candidate_slot = 1; candidate_slot < static_cast<int>(GPU_ROUTE_BRANCH_FANOUT); ++candidate_slot) {
                            if (child_scores[candidate_slot] < child_scores[worst_slot]) {
                                worst_slot = candidate_slot;
                            }
                        }
                        if (child_score > child_scores[worst_slot]) {
                            child_indices[worst_slot] = child_index;
                            child_scores[worst_slot] = child_score;
                        }
                        if (child_role == GALAXY_ROLE_ANTI_PATTERN) {
                            anti_pattern_signal = 1;
                            if (child_score > best_anti_score) {
                                best_anti_score = child_score;
                                anti_index = child_index;
                            }
                        }
                    }
                }

                if (galaxy_route_policy_flag(node_policy, ROUTE_POLICY_DECOMPOSE_ON_FAIL)) {
                    for (unsigned int slot = 0u; slot < 8u; ++slot) {
                        const unsigned int child_index = nearest_indices[slot];
                        if (child_index == 0xFFFFFFFFu) {
                            continue;
                        }
                        const unsigned long long child_hash = galaxy_read_hash(galaxy_table, child_index);
                        bool duplicate = false;
                        for (unsigned int visited_slot = 0u; visited_slot < visited_count; ++visited_slot) {
                            if (visited_hashes[visited_slot] == child_hash) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (duplicate) {
                            continue;
                        }
                        const unsigned int child_role = galaxy_read_selection_role(galaxy_table, child_index);
                        const unsigned int child_family_id = galaxy_read_route_family_id(galaxy_table, child_index);
                        if (!family_candidate_allowed_device(task_family_id, child_family_id)) {
                            continue;
                        }
                        if (child_role != GALAXY_ROLE_ROUTER && child_role != GALAXY_ROLE_EXECUTOR) {
                            continue;
                        }
                        const float child_score =
                            nearest_scores[slot]
                            + route_transition_bonus_device(task_family_id, node_role, child_role)
                            + 0.06f;
                        int worst_slot = 0;
                        for (int candidate_slot = 1; candidate_slot < static_cast<int>(GPU_ROUTE_BRANCH_FANOUT); ++candidate_slot) {
                            if (child_scores[candidate_slot] < child_scores[worst_slot]) {
                                worst_slot = candidate_slot;
                            }
                        }
                        if (child_score > child_scores[worst_slot]) {
                            child_indices[worst_slot] = child_index;
                            child_scores[worst_slot] = child_score;
                        }
                    }
                }

                for (unsigned int child_pick = 0u; child_pick < branch_topk && queue_tail < GPU_ROUTE_FRONTIER_WIDTH; ++child_pick) {
                    int best_child_slot = -1;
                    for (unsigned int candidate_slot = 0u; candidate_slot < GPU_ROUTE_BRANCH_FANOUT; ++candidate_slot) {
                        if (child_indices[candidate_slot] == 0xFFFFFFFFu) {
                            continue;
                        }
                        if (best_child_slot < 0 || child_scores[candidate_slot] > child_scores[best_child_slot]) {
                            best_child_slot = static_cast<int>(candidate_slot);
                        }
                    }
                    if (best_child_slot < 0) {
                        break;
                    }
                    const unsigned int child_index = child_indices[best_child_slot];
                    child_indices[best_child_slot] = 0xFFFFFFFFu;
                    const unsigned long long child_hash = galaxy_read_hash(galaxy_table, child_index);
                    bool duplicate = false;
                    for (unsigned int visited_slot = 0u; visited_slot < visited_count; ++visited_slot) {
                        if (visited_hashes[visited_slot] == child_hash) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) {
                        continue;
                    }
                    const unsigned int child_role = galaxy_read_selection_role(galaxy_table, child_index);
                    const unsigned int child_queue_slot = queue_tail++;
                    queue_indices[child_queue_slot] = child_index;
                    queue_scores[child_queue_slot] = child_scores[best_child_slot];
                    queue_depths[child_queue_slot] = node_depth + 1u;
                    queue_path_lengths[child_queue_slot] = node_path_len < GPU_ROUTE_TRACE_LIMIT ? (node_path_len + 1u) : GPU_ROUTE_TRACE_LIMIT;
                    for (unsigned int copy_slot = 0u; copy_slot < GPU_ROUTE_TRACE_LIMIT; ++copy_slot) {
                        queue_path_indices[child_queue_slot][copy_slot] = queue_path_indices[node_slot][copy_slot];
                        queue_path_roles[child_queue_slot][copy_slot] = queue_path_roles[node_slot][copy_slot];
                    }
                    const unsigned int append_slot = queue_path_lengths[child_queue_slot] - 1u;
                    queue_path_indices[child_queue_slot][append_slot] = child_index;
                    queue_path_roles[child_queue_slot][append_slot] = child_role;
                    if (visited_count < GPU_ROUTE_VISITED_LIMIT) {
                        visited_hashes[visited_count++] = child_hash;
                    }
                }

                if (winner_index != 0xFFFFFFFFu && route_budget_used >= route_budget_min && best_winner_path_len >= 2u) {
                    break;
                }
            }

            const unsigned int* chosen_path = best_winner_path_len > 0u ? best_winner_path
                : (best_validator_path_len > 0u ? best_validator_path
                : (best_executor_path_len > 0u ? best_executor_path : best_router_path));
            const unsigned int* chosen_roles = best_winner_path_len > 0u ? best_winner_roles
                : (best_validator_path_len > 0u ? best_validator_roles
                : (best_executor_path_len > 0u ? best_executor_roles : best_router_roles));
            const unsigned int chosen_path_len = best_winner_path_len > 0u ? best_winner_path_len
                : (best_validator_path_len > 0u ? best_validator_path_len
                : (best_executor_path_len > 0u ? best_executor_path_len : best_router_path_len));

            if (winner_index == 0xFFFFFFFFu) {
                if (validator_index != 0xFFFFFFFFu) {
                    winner_index = validator_index;
                } else if (executor_index != 0xFFFFFFFFu) {
                    winner_index = executor_index;
                } else {
                    winner_index = router_index;
                }
                winner_role = (winner_index != 0xFFFFFFFFu)
                    ? galaxy_read_selection_role(galaxy_table, winner_index)
                    : GALAXY_ROLE_UNKNOWN;
            }

            route_depth = chosen_path_len;
            if (chosen_path_len > recursion_depth_used) {
                recursion_depth_used = chosen_path_len;
            }
            for (unsigned int slot = 0u; slot < GPU_ROUTE_TRACE_LIMIT; ++slot) {
                route_trace_star_indices[slot] = (slot < chosen_path_len) ? chosen_path[slot] : 0xFFFFFFFFu;
                route_trace_role_ids[slot] = (slot < chosen_path_len) ? chosen_roles[slot] : GALAXY_ROLE_UNKNOWN;
            }

            if (router_index == 0xFFFFFFFFu || executor_index == 0xFFFFFFFFu || validator_index == 0xFFFFFFFFu) {
                for (unsigned int slot = 0u; slot < chosen_path_len; ++slot) {
                    const unsigned int role_id = chosen_roles[slot];
                    const unsigned int star_index = chosen_path[slot];
                    if (role_id == GALAXY_ROLE_ROUTER && router_index == 0xFFFFFFFFu) {
                        router_index = star_index;
                    } else if (role_id == GALAXY_ROLE_EXECUTOR) {
                        executor_index = star_index;
                    } else if ((role_id == GALAXY_ROLE_VALIDATOR || role_id == GALAXY_ROLE_ANSWER) && validator_index == 0xFFFFFFFFu) {
                        validator_index = star_index;
                    }
                }
            }

            for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                galaxy_knowledge[index] = 0.0f;
            }
            float total_weight = 0.0f;
            for (unsigned int slot = 0u; slot < chosen_path_len; ++slot) {
                const unsigned int star_index = chosen_path[slot];
                const unsigned int role_id = chosen_roles[slot];
                if (star_index == 0xFFFFFFFFu) {
                    continue;
                }
                float role_weight = 0.18f;
                if (role_id == GALAXY_ROLE_EXECUTOR) {
                    role_weight = 0.30f;
                } else if (role_id == GALAXY_ROLE_VALIDATOR) {
                    role_weight = 0.24f;
                } else if (role_id == GALAXY_ROLE_ANSWER) {
                    role_weight = 0.34f;
                } else if (role_id == GALAXY_ROLE_ANTI_PATTERN) {
                    continue;
                }
                role_weight += 0.02f * static_cast<float>(slot);
                total_weight += role_weight;
                float route_embedding[GPU_TASK_EMBED_DIMS];
                galaxy_compose_embedding_device(route_embedding, galaxy_table, adjacency, star_index, GPU_TASK_EMBED_DIMS);
                for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                    galaxy_knowledge[index] += role_weight * route_embedding[index];
                }
            }
            if (total_weight > 1.0e-6f) {
                const float inv_total = 1.0f / total_weight;
                for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                    galaxy_knowledge[index] *= inv_total;
                }
            }
        }
        __syncthreads();

        if (trm_enabled != 0u) {
            trm_project_query_device(query_embedding, galaxy_knowledge, trm_q, threadIdx.x, blockDim.x);
            for (int index = threadIdx.x; index < GPU_TASK_TRM_DIMS; index += blockDim.x) {
                if (has_brain && frame_count > 0u) {
                    trm_y[index] = brain_trm_y[index];
                    trm_z[index] = brain_trm_z[index];
                } else {
                    trm_y[index] = 0.0f;
                    trm_z[index] = 0.0f;
                }
            }
            __syncthreads();

            const int trm_budget = active_ternary_signal < 0 ? 20 : (active_ternary_signal > 0 ? 8 : 12);
            int trm_steps = 0;
            float trm_drift = 0.0f;
            trm_recursive_core_device(
                trm_q,
                trm_y,
                trm_z,
                W1,
                W2,
                W3,
                W4,
                trm_workspace,
                static_cast<int>(threadIdx.x),
                static_cast<int>(blockDim.x),
                trm_budget,
                1.0e-4f,
                &trm_steps_counter_shared,
                &trm_drift_value,
                &trm_steps,
                &trm_drift
            );
            __syncthreads();

            if (threadIdx.x == 0) {
                trm_steps_used = static_cast<unsigned int>(trm_steps > 0 ? trm_steps : 0);
                trm_drift_value = trm_drift;
            }
            __syncthreads();

            trm_project_latent_to_reasoning_device(
                trm_y,
                trm_z,
                trm_reasoning,
                static_cast<int>(threadIdx.x),
                static_cast<int>(blockDim.x)
            );
            __syncthreads();
            for (int index = threadIdx.x; index < GPU_TASK_EMBED_DIMS; index += blockDim.x) {
                swarm_output[index] = tanhf((0.62f * swarm_output[index]) + (0.38f * trm_reasoning[index]));
            }
            __syncthreads();
        }

        if (family_is_game_like(task_family_id)) {
            blend_with_galaxy_device(
                swarm_output,
                galaxy_knowledge,
                chain_states,
                query_embedding,
                GPU_TASK_EMBED_DIMS,
                0.58f,
                0.28f,
                0.10f,
                0.04f,
                3u,
                2u
            );
            geometry_route_device(swarm_output, galaxy_knowledge, GPU_TASK_EMBED_DIMS);
            fractal_emit_device(swarm_output, galaxy_knowledge, GPU_TASK_EMBED_DIMS);
        } else if (family_is_math_like(task_family_id)) {
            blend_with_galaxy_device(
                swarm_output,
                galaxy_knowledge,
                chain_states,
                query_embedding,
                GPU_TASK_EMBED_DIMS,
                0.48f,
                0.36f,
                0.12f,
                0.04f,
                0u,
                4u
            );
            atomic_fission_fusion_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
            temporal_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
        } else if (family_is_question_like(task_family_id)) {
            blend_with_galaxy_device(
                swarm_output,
                galaxy_knowledge,
                chain_states,
                query_embedding,
                GPU_TASK_EMBED_DIMS,
                0.50f,
                0.34f,
                0.10f,
                0.06f,
                0u,
                5u
            );
            graph_crystallize_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
            temporal_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
        } else {
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
        }
        cognitive_executive_device(resonance_scores, chain_states, swarm_output, GPU_TASK_EMBED_DIMS);

        if (threadIdx.x == 0) {
            iterations_used = think_step + 1u;
            if (bounded_options == 0u) {
                best_index = 0u;
                best_score = winner_index == 0xFFFFFFFFu
                    ? 0.0f
                    : star_selection_score_device(galaxy_table, adjacency, winner_index, swarm_output, task_family_id);
                converged = (winner_index != 0xFFFFFFFFu) ? 1 : 0;
            } else {
                float winner_embedding[GPU_TASK_EMBED_DIMS];
                if (winner_index != 0xFFFFFFFFu) {
                    galaxy_compose_embedding_device(winner_embedding, galaxy_table, adjacency, winner_index, GPU_TASK_EMBED_DIMS);
                } else {
                    for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                        winner_embedding[index] = galaxy_knowledge[index];
                    }
                }
                for (unsigned int option_index = 0u; option_index < bounded_options; ++option_index) {
                    const float* option_embedding = option_embeddings + (option_index * GPU_TASK_EMBED_DIMS);
                    float score = cosine32_device(swarm_output, option_embedding, GPU_TASK_EMBED_DIMS);
                    score += 0.15f * cosine32_device(winner_embedding, option_embedding, GPU_TASK_EMBED_DIMS);
                    if (family_is_game_like(task_family_id)) {
                        const unsigned char* active_action_history =
                            (has_brain && frame_count > 0u) ? brain_action_ring : action_history;
                        for (unsigned int history_index = 0u; history_index < active_action_history_len; ++history_index) {
                            if (static_cast<unsigned int>(active_action_history[history_index]) == option_index) {
                                const float recency = 0.25f + (0.12f * static_cast<float>(history_index));
                                score *= recency;
                                break;
                            }
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
                const float min_threshold = family_is_question_like(task_family_id) ? 0.04f : (family_is_math_like(task_family_id) ? 0.06f : 0.10f);
                const float gap_threshold = family_is_question_like(task_family_id) ? 0.04f : (family_is_math_like(task_family_id) ? 0.08f : 0.12f);
                converged = halting_gate_device(
                    resonance_scores,
                    GPU_TASK_NUM_CHAINS,
                    min_threshold,
                    gap_threshold,
                    0.7f
                );
                if (route_budget_used < route_budget_min || recursion_depth_used < 2u) {
                    converged = 0;
                }
                if (expected_index >= 0 && static_cast<unsigned int>(expected_index) == best_index) {
                    converged = 1;
                }
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
        for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
            if (device_absf(goal_embedding[index]) > 1.0e-8f) {
                has_goal_embedding = true;
                break;
            }
        }
        goal_progress = has_goal_embedding
            ? goal_progress_device(reasoning_state, goal_embedding, query_embedding, GPU_TASK_EMBED_DIMS)
            : 0.0f;

        const unsigned long long answer_hash =
            (bounded_options > 0u && best_index < bounded_options)
                ? option_hashes[best_index]
                : ((winner_index != 0xFFFFFFFFu) ? galaxy_read_hash(galaxy_table, winner_index) : 0ull);

        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ANSWER_INDEX_OUTPUT_OFFSET) = best_index;
        *reinterpret_cast<float*>(output_buffer + output_base + GPU_TASK_CONFIDENCE_OUTPUT_OFFSET) = best_score;
        *reinterpret_cast<signed char*>(output_buffer + output_base + GPU_TASK_CONVERGENCE_OUTPUT_OFFSET) =
            bounded_options > 0u ? static_cast<signed char>(converged ? 1 : 0) : static_cast<signed char>(winner_index != 0xFFFFFFFFu ? 1 : 0);
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ITERATIONS_OUTPUT_OFFSET) = iterations_used;
        *reinterpret_cast<unsigned long long*>(output_buffer + output_base + GPU_TASK_ANSWER_HASH_OUTPUT_OFFSET) = answer_hash;
        *reinterpret_cast<float*>(output_buffer + output_base + GPU_TASK_GOAL_PROGRESS_OUTPUT_OFFSET) = goal_progress;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_WINNER_STAR_INDEX_OUTPUT_OFFSET) = winner_index;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_WINNER_ROLE_ID_OUTPUT_OFFSET) = winner_role;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ROUTE_DEPTH_OUTPUT_OFFSET) = route_depth;
        *reinterpret_cast<int*>(output_buffer + output_base + GPU_TASK_ANTI_PATTERN_SIGNAL_OUTPUT_OFFSET) = anti_pattern_signal;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ROUTER_STAR_INDEX_OUTPUT_OFFSET) = router_index;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_EXECUTOR_STAR_INDEX_OUTPUT_OFFSET) = executor_index;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_VALIDATOR_STAR_INDEX_OUTPUT_OFFSET) = validator_index;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ROUTE_BUDGET_USED_OUTPUT_OFFSET) = route_budget_used;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ROUTE_BUDGET_MIN_OUTPUT_OFFSET) = route_budget_min;
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_RECURSION_DEPTH_USED_OUTPUT_OFFSET) = recursion_depth_used;
        for (unsigned int slot = 0u; slot < GPU_ROUTE_TRACE_LIMIT; ++slot) {
            *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ROUTE_TRACE_STAR_INDICES_OUTPUT_OFFSET + (slot * 4u)) =
                route_trace_star_indices[slot];
            *reinterpret_cast<unsigned int*>(output_buffer + output_base + GPU_TASK_ROUTE_TRACE_ROLE_IDS_OUTPUT_OFFSET + (slot * 4u)) =
                route_trace_role_ids[slot];
        }

        if (lesson_buffer != nullptr && lesson_counter != nullptr && lesson_capacity > 0u) {
            const unsigned int lesson_slot = atomicAdd(lesson_counter, 1u) % lesson_capacity;
            const unsigned int lesson_base = lesson_slot * LESSON_RECORD_BYTES;
            const unsigned long long trace_hash = route_trace_hash_device(
                route_trace_star_indices,
                route_trace_role_ids,
                route_depth
            );
            *reinterpret_cast<unsigned int*>(lesson_buffer + lesson_base + LESSON_FAMILY_ID_OFFSET) = task_family_id;
            *reinterpret_cast<unsigned int*>(lesson_buffer + lesson_base + LESSON_ROUTER_INDEX_OFFSET) = router_index;
            *reinterpret_cast<unsigned int*>(lesson_buffer + lesson_base + LESSON_EXECUTOR_INDEX_OFFSET) = executor_index;
            *reinterpret_cast<unsigned int*>(lesson_buffer + lesson_base + LESSON_VALIDATOR_INDEX_OFFSET) = validator_index;
            *reinterpret_cast<unsigned int*>(lesson_buffer + lesson_base + LESSON_WINNER_INDEX_OFFSET) = winner_index;
            *reinterpret_cast<unsigned int*>(lesson_buffer + lesson_base + LESSON_WINNER_ROLE_OFFSET) = winner_role;
            *reinterpret_cast<unsigned long long*>(lesson_buffer + lesson_base + LESSON_EXPECTED_HASH_OFFSET) = expected_hash;
            *reinterpret_cast<unsigned long long*>(lesson_buffer + lesson_base + LESSON_PREDICTED_HASH_OFFSET) = answer_hash;
            float reward = 0.0f;
            if (expected_hash != 0ull && answer_hash != 0ull) {
                reward = (expected_hash == answer_hash) ? 1.0f : -1.0f;
            } else if (expected_index >= 0 && bounded_options > 0u) {
                reward = (static_cast<unsigned int>(expected_index) == best_index) ? 1.0f : -1.0f;
            }
            *reinterpret_cast<float*>(lesson_buffer + lesson_base + LESSON_REWARD_OFFSET) = reward;
            *reinterpret_cast<int*>(lesson_buffer + lesson_base + LESSON_ANTI_PATTERN_OFFSET) = anti_pattern_signal;
            *reinterpret_cast<unsigned int*>(lesson_buffer + lesson_base + LESSON_ROUTE_DEPTH_OFFSET) = route_depth;
            *reinterpret_cast<unsigned long long*>(lesson_buffer + lesson_base + LESSON_ROUTE_TRACE_HASH_OFFSET) = trace_hash;
        }
    }

    if (has_brain) {
        for (int index = threadIdx.x; index < GPU_TASK_EMBED_DIMS; index += blockDim.x) {
            brain_reasoning[index] = reasoning_state[index];
            brain_prev_frame[index] = query_embedding[index];
        }
        for (int index = threadIdx.x; index < GPU_TASK_TRM_DIMS; index += blockDim.x) {
            brain_trm_q[index] = trm_q[index];
            brain_trm_y[index] = trm_y[index];
            brain_trm_z[index] = trm_z[index];
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
            const int learned_ternary = family_is_game_like(task_family_id)
                ? static_cast<int>(quantize_trit_device(goal_progress))
                : active_ternary_signal;
            *reinterpret_cast<signed char*>(brain_state + BRAIN_TERNARY_OFFSET) =
                static_cast<signed char>(learned_ternary);
            *reinterpret_cast<unsigned int*>(brain_state + BRAIN_FRAME_COUNT_OFFSET) = frame_count + 1u;
        }
    }
}
