typedef int int32_t;

#define KNN_BLOCK_THREADS 128
#define MAX_TOP_K 32

__device__ __forceinline__ void insert_top_k(
    float score,
    int32_t index,
    int32_t effective_k,
    float* best_scores,
    int32_t* best_indices
) {
    int32_t insert_at = -1;
    for (int32_t slot = 0; slot < effective_k; ++slot) {
        if (score > best_scores[slot]) {
            insert_at = slot;
            break;
        }
    }
    if (insert_at < 0) {
        return;
    }
    for (int32_t slot = effective_k - 1; slot > insert_at; --slot) {
        best_scores[slot] = best_scores[slot - 1];
        best_indices[slot] = best_indices[slot - 1];
    }
    best_scores[insert_at] = score;
    best_indices[insert_at] = index;
}

extern "C" __global__ void knn_graph_build(
    const float* __restrict__ embeddings,
    int32_t total_entries,
    int32_t dim,
    int32_t source_offset,
    int32_t source_count,
    int32_t top_k,
    float similarity_threshold,
    int32_t* __restrict__ out_neighbors,
    float* __restrict__ out_similarities,
    int32_t* __restrict__ out_counts
) {
    const int32_t local_row = (int32_t)blockIdx.x;
    const int32_t source_index = source_offset + local_row;
    const int32_t effective_k = top_k < MAX_TOP_K ? top_k : MAX_TOP_K;
    if (local_row >= source_count || source_index >= total_entries) {
        return;
    }

    if (effective_k <= 0) {
        if (threadIdx.x == 0) {
            out_counts[local_row] = 0;
        }
        return;
    }

    __shared__ float shared_scores[KNN_BLOCK_THREADS * MAX_TOP_K];
    __shared__ int32_t shared_indices[KNN_BLOCK_THREADS * MAX_TOP_K];

    float thread_scores[MAX_TOP_K];
    int32_t thread_indices[MAX_TOP_K];
    for (int32_t slot = 0; slot < effective_k; ++slot) {
        thread_scores[slot] = -3.402823466e+38f;
        thread_indices[slot] = -1;
    }

    const float* source_row = embeddings + ((long long)source_index * dim);
    for (int32_t target_index = (int32_t)threadIdx.x; target_index < total_entries; target_index += blockDim.x) {
        if (target_index == source_index) {
            continue;
        }
        const float* target_row = embeddings + ((long long)target_index * dim);
        float similarity = 0.0f;
        for (int32_t col = 0; col < dim; ++col) {
            similarity += source_row[col] * target_row[col];
        }
        if (similarity < similarity_threshold) {
            continue;
        }
        insert_top_k(similarity, target_index, effective_k, thread_scores, thread_indices);
    }

    const int32_t shared_base = ((int32_t)threadIdx.x) * MAX_TOP_K;
    for (int32_t slot = 0; slot < effective_k; ++slot) {
        shared_scores[shared_base + slot] = thread_scores[slot];
        shared_indices[shared_base + slot] = thread_indices[slot];
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float final_scores[MAX_TOP_K];
    int32_t final_indices[MAX_TOP_K];
    for (int32_t slot = 0; slot < effective_k; ++slot) {
        final_scores[slot] = -3.402823466e+38f;
        final_indices[slot] = -1;
    }

    for (int32_t worker = 0; worker < blockDim.x; ++worker) {
        const int32_t worker_base = worker * MAX_TOP_K;
        for (int32_t slot = 0; slot < effective_k; ++slot) {
            const int32_t candidate_index = shared_indices[worker_base + slot];
            if (candidate_index < 0) {
                continue;
            }
            insert_top_k(
                shared_scores[worker_base + slot],
                candidate_index,
                effective_k,
                final_scores,
                final_indices
            );
        }
    }

    const int32_t row_base = local_row * top_k;
    int32_t found = 0;
    for (int32_t slot = 0; slot < top_k; ++slot) {
        out_neighbors[row_base + slot] = -1;
        out_similarities[row_base + slot] = -3.402823466e+38f;
    }
    for (int32_t slot = 0; slot < effective_k; ++slot) {
        if (final_indices[slot] < 0) {
            continue;
        }
        out_neighbors[row_base + found] = final_indices[slot];
        out_similarities[row_base + found] = final_scores[slot];
        found += 1;
    }
    out_counts[local_row] = found;
}
