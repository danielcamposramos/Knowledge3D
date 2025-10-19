/*
 * Glyph Resonator - Phase C3 implementation.
 * Performs cosine-similarity matching between extracted character features
 * and the learned glyph embeddings harvested in Phase B.
 */

#include <cuda_runtime.h>
#include <math.h>

#define EMBEDDING_DIM 128

__device__ inline float cosine_similarity(const float* a, const float* b) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        const float va = a[i];
        const float vb = b[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    if (norm_a < 1e-8f || norm_b < 1e-8f) {
        return -1.0f;
    }

    return dot / (norm_a * norm_b);
}

extern "C" __global__ void glyph_resonator(
    float* output,                 // [num_chars * 3] -> char_idx, glyph_idx, confidence
    const float* char_features,    // [num_chars * 128]
    int num_chars,
    const float* glyph_embeddings, // [glyph_count * 128]
    int glyph_count
) {
    const int char_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (char_idx >= num_chars) {
        return;
    }

    const float* feature = char_features + (char_idx * EMBEDDING_DIM);

    float best_score = -1.0f;
    int best_glyph = -1;

    for (int glyph_idx = 0; glyph_idx < glyph_count; ++glyph_idx) {
        const float* glyph = glyph_embeddings + (glyph_idx * EMBEDDING_DIM);
        const float score = cosine_similarity(feature, glyph);
        if (score > best_score) {
            best_score = score;
            best_glyph = glyph_idx;
        }
    }

    const int out_offset = char_idx * 3;
    output[out_offset + 0] = static_cast<float>(char_idx);
    output[out_offset + 1] = static_cast<float>(best_glyph);
    output[out_offset + 2] = best_score;
}
