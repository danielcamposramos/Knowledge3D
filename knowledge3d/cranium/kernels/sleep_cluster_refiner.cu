#include <math.h>

extern "C" __global__ void assign_to_best_centroid(
    const float* similarities,    // [N x k]
    int* assignments,             // [N]
    int N,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (k <= 0) {
        assignments[idx] = -1;
        return;
    }

    const float* row = similarities + (idx * k);
    float best_score = row[0];
    int best_cluster = 0;
    for (int cluster_id = 1; cluster_id < k; ++cluster_id) {
        float score = row[cluster_id];
        if (score > best_score) {
            best_score = score;
            best_cluster = cluster_id;
        }
    }

    assignments[idx] = best_cluster;
}

extern "C" __global__ void accumulate_centroid_sums(
    const float* embeddings,      // [N x dim]
    const int* assignments,       // [N]
    float* centroid_sums,         // [k x dim]
    int* counts,                  // [k]
    int N,
    int dim,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * dim;
    if (idx >= total) return;

    int row = idx / dim;
    int col = idx % dim;
    int cluster_id = assignments[row];
    if (cluster_id < 0 || cluster_id >= k) return;

    atomicAdd(centroid_sums + (cluster_id * dim) + col, embeddings[idx]);
    if (col == 0) {
        atomicAdd(counts + cluster_id, 1);
    }
}

extern "C" __global__ void finalize_centroids(
    const float* centroid_sums,   // [k x dim]
    const int* counts,            // [k]
    float* centroids,             // [k x dim]
    int k,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * dim;
    if (idx >= total) return;

    int cluster_id = idx / dim;
    int count = counts[cluster_id];
    if (count <= 0) {
        centroids[idx] = 0.0f;
        return;
    }

    const float* cluster_sums = centroid_sums + (cluster_id * dim);
    float norm_sq = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float mean_val = cluster_sums[d] / (float)count;
        norm_sq += mean_val * mean_val;
    }

    float norm = sqrtf(norm_sq);
    float value = cluster_sums[idx % dim] / (float)count;
    if (norm > 1e-8f) {
        value /= norm;
    } else {
        value = 0.0f;
    }
    centroids[idx] = value;
}

extern "C" __global__ void refine_embeddings_to_centroids(
    float* embeddings,          // [N x dim] updated in place
    int N,
    int dim,
    const float* centroids,     // [k x dim]
    const int* assignments,     // [N]
    int k,
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int cluster_id = assignments[idx];
    if (cluster_id < 0 || cluster_id >= k) return;

    float* embedding = embeddings + (idx * dim);
    const float* centroid = centroids + (cluster_id * dim);

    float norm_sq = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float updated = embedding[d] + learning_rate * (centroid[d] - embedding[d]);
        embedding[d] = updated;
        norm_sq += updated * updated;
    }

    float norm = sqrtf(norm_sq);
    if (norm > 1e-8f) {
        float inv_norm = 1.0f / norm;
        for (int d = 0; d < dim; ++d) {
            embedding[d] *= inv_norm;
        }
    }
}

extern "C" __global__ void compute_silhouette_scores(
    float* scores,              // [N]
    const float* embeddings,    // [N x dim]
    const int* assignments,     // [N]
    int N,
    int dim,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* emb_i = embeddings + (idx * dim);
    int cluster_i = assignments[idx];
    if (cluster_i < 0 || cluster_i >= k) {
        scores[idx] = 0.0f;
        return;
    }

    float sum_intra = 0.0f;
    int count_intra = 0;

    for (int j = 0; j < N; ++j) {
        if (j == idx) continue;
        if (assignments[j] != cluster_i) continue;
        const float* emb_j = embeddings + (j * dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += emb_i[d] * emb_j[d];
        }
        sum_intra += 1.0f - dot;
        count_intra++;
    }

    float a = (count_intra > 0) ? (sum_intra / count_intra) : 0.0f;
    float b = 2.0f;

    for (int other_cluster = 0; other_cluster < k; ++other_cluster) {
        if (other_cluster == cluster_i) continue;

        float sum_inter = 0.0f;
        int count_inter = 0;
        for (int j = 0; j < N; ++j) {
            if (assignments[j] != other_cluster) continue;
            const float* emb_j = embeddings + (j * dim);
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += emb_i[d] * emb_j[d];
            }
            sum_inter += 1.0f - dot;
            count_inter++;
        }

        if (count_inter > 0) {
            float mean_dist = sum_inter / count_inter;
            if (mean_dist < b) b = mean_dist;
        }
    }

    float max_ab = fmaxf(a, b);
    scores[idx] = (max_ab > 1e-8f) ? ((b - a) / max_ab) : 0.0f;
}
