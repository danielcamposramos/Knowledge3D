// Vector Resonator - attention-weighted blending over a small vector set.
//
// Inputs:
//   vectors  [K x D] flattened row-major input vectors
// Outputs:
//   blended  [D]     attention-weighted blend
//   weights  [K]     per-vector attention weights
//
// The hot path typically runs with K=2 and D=16. We serialize the attention
// computation in thread 0 because K is intentionally tiny, and correctness is
// more important than occupancy for this post-swarm refinement stage.

#include <math.h>

extern "C" __global__ void gre_vector_resonator(
    const float* __restrict__ vectors,
    float* __restrict__ blended,
    float* __restrict__ attention_weights,
    int K,
    int D
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (K <= 0 || D <= 0) {
        return;
    }

    const int kMaxVectors = 32;
    if (K > kMaxVectors) {
        K = kMaxVectors;
    }

    float norms[kMaxVectors];
    float scores[kMaxVectors];
    float weights[kMaxVectors];
    const float eps = 1e-12f;

    for (int d = 0; d < D; ++d) {
        blended[d] = 0.0f;
    }

    for (int k = 0; k < K; ++k) {
        float energy_sq = 0.0f;
        const int base = k * D;
        for (int d = 0; d < D; ++d) {
            const float value = vectors[base + d];
            energy_sq += value * value;
        }
        norms[k] = sqrtf(energy_sq + eps);
    }

    for (int k = 0; k < K; ++k) {
        const int base = k * D;
        const float energy = norms[k];
        float cross = 0.0f;
        if (K > 1) {
            for (int j = 0; j < K; ++j) {
                if (j == k) {
                    continue;
                }
                float dot = 0.0f;
                const int other = j * D;
                for (int d = 0; d < D; ++d) {
                    dot += vectors[base + d] * vectors[other + d];
                }
                const float cosine = dot / (energy * norms[j] + eps);
                cross += cosine;
            }
            cross /= (float)(K - 1);
        }
        scores[k] = energy * (1.0f + fmaxf(cross, 0.0f));
    }

    float max_score = scores[0];
    for (int k = 1; k < K; ++k) {
        if (scores[k] > max_score) {
            max_score = scores[k];
        }
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float weight = expf(scores[k] - max_score);
        weights[k] = weight;
        sum_exp += weight;
    }

    const float inv_sum = 1.0f / (sum_exp + eps);
    for (int k = 0; k < K; ++k) {
        const float normalized = weights[k] * inv_sum;
        attention_weights[k] = normalized;
        const int base = k * D;
        for (int d = 0; d < D; ++d) {
            blended[d] += normalized * vectors[base + d];
        }
    }
}
