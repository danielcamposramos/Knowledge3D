typedef int int32_t;
typedef unsigned int uint32_t;

#define GRAPH_QUEUE_CAPACITY 4096

static __device__ float unpack_edge_cost(uint32_t packed, float alpha, float beta) {
    const float geo = (float)(packed & 0xFFFFu);
    const float sem = (float)((packed >> 16) & 0xFFFFu);
    return (alpha * geo) + (beta * sem);
}

static __device__ bool contains_node(const int32_t* nodes, int32_t count, int32_t value) {
    for (int32_t idx = 0; idx < count; ++idx) {
        if (nodes[idx] == value) {
            return true;
        }
    }
    return false;
}

static __device__ int32_t find_node_index(const int32_t* nodes, int32_t count, int32_t value) {
    for (int32_t idx = 0; idx < count; ++idx) {
        if (nodes[idx] == value) {
            return idx;
        }
    }
    return -1;
}

extern "C" __global__ void graph_expand_bfs(
    const uint32_t* __restrict__ row_offsets,
    const uint32_t* __restrict__ col_indices,
    const uint32_t* __restrict__ packed_costs,
    const int32_t* __restrict__ seed_indices,
    int32_t seed_count,
    int32_t max_nodes,
    int32_t max_edge_expansions,
    float alpha,
    float beta,
    int32_t* __restrict__ selected_nodes,
    int32_t* __restrict__ selected_count,
    uint32_t* __restrict__ local_row_offsets,
    uint32_t* __restrict__ local_col_indices,
    uint32_t* __restrict__ local_packed_costs,
    int32_t* __restrict__ local_edge_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (seed_count <= 0 || max_nodes <= 0) {
        *selected_count = 0;
        local_row_offsets[0] = 0u;
        *local_edge_count = 0;
        return;
    }

    int32_t frontier_nodes[GRAPH_QUEUE_CAPACITY];
    float frontier_costs[GRAPH_QUEUE_CAPACITY];
    int32_t frontier_count = 0;

    for (int32_t idx = 0; idx < seed_count && frontier_count < GRAPH_QUEUE_CAPACITY; ++idx) {
        const int32_t node = seed_indices[idx];
        if (node < 0 || contains_node(frontier_nodes, frontier_count, node)) {
            continue;
        }
        frontier_nodes[frontier_count] = node;
        frontier_costs[frontier_count] = 0.0f;
        frontier_count += 1;
    }

    int32_t chosen_count = 0;
    int32_t expansions = 0;

    while (frontier_count > 0 && chosen_count < max_nodes && expansions < max_edge_expansions) {
        int32_t best_frontier = 0;
        float best_cost = frontier_costs[0];
        for (int32_t idx = 1; idx < frontier_count; ++idx) {
            if (frontier_costs[idx] < best_cost) {
                best_cost = frontier_costs[idx];
                best_frontier = idx;
            }
        }

        const int32_t current_node = frontier_nodes[best_frontier];
        frontier_count -= 1;
        frontier_nodes[best_frontier] = frontier_nodes[frontier_count];
        frontier_costs[best_frontier] = frontier_costs[frontier_count];

        if (contains_node(selected_nodes, chosen_count, current_node)) {
            continue;
        }

        selected_nodes[chosen_count] = current_node;
        chosen_count += 1;

        const uint32_t row_start = row_offsets[current_node];
        const uint32_t row_end = row_offsets[current_node + 1];
        for (uint32_t edge_idx = row_start; edge_idx < row_end; ++edge_idx) {
            const int32_t neighbor = (int32_t)(col_indices[edge_idx]);
            if (neighbor < 0 || contains_node(selected_nodes, chosen_count, neighbor)) {
                continue;
            }
            const float tentative_cost = best_cost + unpack_edge_cost(packed_costs[edge_idx], alpha, beta);
            const int32_t existing_frontier = find_node_index(frontier_nodes, frontier_count, neighbor);
            if (existing_frontier >= 0) {
                if (tentative_cost < frontier_costs[existing_frontier]) {
                    frontier_costs[existing_frontier] = tentative_cost;
                }
            } else if (frontier_count < GRAPH_QUEUE_CAPACITY) {
                frontier_nodes[frontier_count] = neighbor;
                frontier_costs[frontier_count] = tentative_cost;
                frontier_count += 1;
            }
            expansions += 1;
            if (expansions >= max_edge_expansions) {
                break;
            }
        }
    }

    *selected_count = chosen_count;
    uint32_t edge_out = 0u;
    local_row_offsets[0] = 0u;
    for (int32_t local_idx = 0; local_idx < chosen_count; ++local_idx) {
        const int32_t global_node = selected_nodes[local_idx];
        const uint32_t row_start = row_offsets[global_node];
        const uint32_t row_end = row_offsets[global_node + 1];
        for (uint32_t edge_idx = row_start; edge_idx < row_end; ++edge_idx) {
            if ((int32_t)edge_out >= max_edge_expansions) {
                break;
            }
            const int32_t global_neighbor = (int32_t)(col_indices[edge_idx]);
            const int32_t local_neighbor = find_node_index(selected_nodes, chosen_count, global_neighbor);
            if (local_neighbor < 0) {
                continue;
            }
            local_col_indices[edge_out] = (uint32_t)local_neighbor;
            local_packed_costs[edge_out] = packed_costs[edge_idx];
            edge_out += 1u;
        }
        local_row_offsets[local_idx + 1] = edge_out;
        if ((int32_t)edge_out >= max_edge_expansions) {
            for (int32_t fill_idx = local_idx + 1; fill_idx < chosen_count; ++fill_idx) {
                local_row_offsets[fill_idx + 1] = edge_out;
            }
            break;
        }
    }
    *local_edge_count = (int32_t)edge_out;
}
