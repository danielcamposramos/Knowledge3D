typedef int int32_t;
typedef unsigned short uint16_t;

#define MAX_TOP_K 64

extern "C" __global__ void seed_select_top_k(
    const float* __restrict__ embeddings,
    const int32_t* __restrict__ galaxy_ids,
    const float* __restrict__ query,
    const int32_t* __restrict__ allowed_galaxies,
    int32_t allowed_count,
    int32_t num_entries,
    int32_t dim,
    int32_t top_k,
    float similarity_threshold,
    int32_t target_cluster_id,
    float cluster_bias,
    const uint16_t* __restrict__ subject_clusters,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_similarities,
    int32_t* __restrict__ out_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const int32_t effective_top_k = top_k < MAX_TOP_K ? top_k : MAX_TOP_K;
    if (effective_top_k <= 0) {
        *out_count = 0;
        return;
    }

    float best_scores[MAX_TOP_K];
    int32_t best_indices[MAX_TOP_K];
    for (int32_t i = 0; i < effective_top_k; ++i) {
        best_scores[i] = -3.402823466e+38f;
        best_indices[i] = -1;
    }

    int32_t found = 0;
    for (int32_t idx = 0; idx < num_entries; ++idx) {
        bool allowed = (allowed_count <= 0);
        if (!allowed) {
            const int32_t galaxy_id = galaxy_ids[idx];
            for (int32_t g = 0; g < allowed_count; ++g) {
                if (allowed_galaxies[g] == galaxy_id) {
                    allowed = true;
                    break;
                }
            }
        }
        if (!allowed) {
            continue;
        }

        const float* row = embeddings + (idx * dim);
        float similarity = 0.0f;
        for (int32_t col = 0; col < dim; ++col) {
            similarity += row[col] * query[col];
        }
        if (
            target_cluster_id > 0
            && subject_clusters != 0
            && (int32_t)(subject_clusters[idx]) == target_cluster_id
        ) {
            similarity += cluster_bias;
        }
        if (similarity < similarity_threshold) {
            continue;
        }

        int32_t insert_at = -1;
        for (int32_t slot = 0; slot < effective_top_k; ++slot) {
            if (similarity > best_scores[slot]) {
                insert_at = slot;
                break;
            }
        }
        if (insert_at < 0) {
            continue;
        }
        for (int32_t slot = effective_top_k - 1; slot > insert_at; --slot) {
            best_scores[slot] = best_scores[slot - 1];
            best_indices[slot] = best_indices[slot - 1];
        }
        best_scores[insert_at] = similarity;
        best_indices[insert_at] = idx;
        if (found < effective_top_k) {
            found += 1;
        }
    }

    for (int32_t slot = 0; slot < effective_top_k; ++slot) {
        out_indices[slot] = best_indices[slot];
        out_similarities[slot] = best_scores[slot];
    }
    *out_count = found;
}
