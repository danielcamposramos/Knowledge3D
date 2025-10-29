/*
 * Glyph Resonator - Matryoshka variant.
 * Matches character feature vectors against multi-scale glyph embeddings.
 *
 * Supports variable-dimensional queries (64D → 2048D) while honouring
 * the Matryoshka prefix property: higher-dimensional glyphs contain the
 * lower-dimensional slices as their leading segments.
 */

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void glyph_resonator_matryoshka(
    float* output,                  // [num_chars * 3] -> char_idx, glyph_idx, confidence
    const float* char_features,     // [num_chars * feature_dim]
    int num_chars,
    const float* glyph_embeddings,  // [glyph_count * max_dim]
    int glyph_count,
    const int* glyph_dims,          // [glyph_count] native dimension per glyph
    int feature_dim,                // Query (character) dimension
    int max_dim                     // Maximum glyph dimension (e.g., 2048)
) {
    const int char_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (char_idx >= num_chars) {
        return;
    }

    const float* feature = char_features + (char_idx * feature_dim);

    float norm_a = 0.0f;
    for (int d = 0; d < feature_dim; ++d) {
        const float va = feature[d];
        norm_a += va * va;
    }

    if (norm_a < 1e-8f) {
        const int out_offset = char_idx * 3;
        output[out_offset + 0] = static_cast<float>(char_idx);
        output[out_offset + 1] = -1.0f;
        output[out_offset + 2] = -1.0f;
        return;
    }

    const float inv_norm_a = rsqrtf(norm_a);

    float best_score = -1.0f;
    int best_glyph = -1;

    for (int glyph_idx = 0; glyph_idx < glyph_count; ++glyph_idx) {
        int glyph_dim = glyph_dims[glyph_idx];
        if (glyph_dim <= 0) {
            continue;
        }
        if (glyph_dim > max_dim) {
            glyph_dim = max_dim;
        }

        const int effective_dim = glyph_dim < feature_dim ? glyph_dim : feature_dim;
        if (effective_dim <= 0) {
            continue;
        }

        const float* glyph = glyph_embeddings + (glyph_idx * max_dim);

        float dot = 0.0f;
        float norm_b = 0.0f;

        for (int d = 0; d < effective_dim; ++d) {
            const float va = feature[d];
            const float vb = glyph[d];
            dot += va * vb;
            norm_b += vb * vb;
        }

        if (norm_b < 1e-8f) {
            continue;
        }

        const float score = dot * inv_norm_a / sqrtf(norm_b);
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

