/*
 * glyph_match.cu - Character Template Matching Kernel
 *
 * Implements fast template matching for OCR character recognition.
 * Uses normalized cross-correlation to match feature maps against
 * learned glyph templates.
 *
 * Architecture:
 *   - Template size: 8×8 pixels (typical character patch)
 *   - Features: 128 channels (from conv stack output)
 *   - Output: Confidence scores for each glyph class
 *   - Method: Normalized cross-correlation with L2 norm
 *
 * Performance target: <50µs per patch on RTX 3060
 */

#include <cuda_runtime.h>

#define MAX_GLYPHS 256  // Support up to 256 character classes
#define TEMPLATE_SIZE 8
#define FEATURE_CHANNELS 128

/*
 * glyph_match_ncc - Normalized Cross-Correlation for glyph matching
 *
 * Grid: (num_patches, 1, 1)
 * Block: (TEMPLATE_SIZE * TEMPLATE_SIZE, 1, 1)
 *
 * Args:
 *   features: Input feature map [num_patches, 8, 8, 128]
 *   templates: Glyph templates [num_glyphs, 8, 8, 128]
 *   scores: Output scores [num_patches, num_glyphs]
 *   num_patches: Number of patches to match
 *   num_glyphs: Number of glyph classes
 */
extern "C" __global__ void glyph_match_ncc(
    const float* __restrict__ features,
    const float* __restrict__ templates,
    float* __restrict__ scores,
    int num_patches,
    int num_glyphs
) {
    const int patch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (patch_idx >= num_patches) return;

    // Shared memory for template and feature patch
    __shared__ float feature_patch[TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS];
    __shared__ float template_cache[TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS];
    __shared__ float partial_sums[256];  // For reduction

    // Load feature patch into shared memory
    const int elements_per_thread = (TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS) /
                                    (TEMPLATE_SIZE * TEMPLATE_SIZE);

    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid * elements_per_thread + i;
        if (idx < TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS) {
            feature_patch[idx] = features[patch_idx * TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS + idx];
        }
    }
    __syncthreads();

    // Compute feature patch L2 norm
    float feature_norm_sq = 0.0f;
    for (int i = tid; i < TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS; i += blockDim.x) {
        float val = feature_patch[i];
        feature_norm_sq += val * val;
    }

    // Warp-level reduction for norm
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        feature_norm_sq += __shfl_down_sync(0xffffffff, feature_norm_sq, offset);
    }

    if (tid == 0) {
        partial_sums[0] = feature_norm_sq;
    }
    __syncthreads();

    float feature_norm = sqrtf(partial_sums[0]);

    // Match against each glyph template
    for (int glyph_idx = 0; glyph_idx < num_glyphs; ++glyph_idx) {
        // Load template into shared memory
        for (int i = 0; i < elements_per_thread; ++i) {
            int idx = tid * elements_per_thread + i;
            if (idx < TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS) {
                template_cache[idx] = templates[glyph_idx * TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS + idx];
            }
        }
        __syncthreads();

        // Compute template norm and dot product
        float template_norm_sq = 0.0f;
        float dot_product = 0.0f;

        for (int i = tid; i < TEMPLATE_SIZE * TEMPLATE_SIZE * FEATURE_CHANNELS; i += blockDim.x) {
            float t = template_cache[i];
            float f = feature_patch[i];
            template_norm_sq += t * t;
            dot_product += t * f;
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            template_norm_sq += __shfl_down_sync(0xffffffff, template_norm_sq, offset);
            dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
        }

        if ((tid & 31) == 0) {  // Lane 0 of each warp
            int warp_idx = tid / 32;
            partial_sums[warp_idx] = template_norm_sq;
            partial_sums[warp_idx + 8] = dot_product;
        }
        __syncthreads();

        // Final reduction by thread 0
        if (tid == 0) {
            float total_template_norm_sq = 0.0f;
            float total_dot = 0.0f;
            for (int i = 0; i < 2; ++i) {  // 64 threads = 2 warps
                total_template_norm_sq += partial_sums[i];
                total_dot += partial_sums[i + 8];
            }

            float template_norm = sqrtf(total_template_norm_sq);

            // Normalized cross-correlation
            float ncc = total_dot / (feature_norm * template_norm + 1e-8f);

            // Convert to confidence score (0-1 range)
            scores[patch_idx * num_glyphs + glyph_idx] = (ncc + 1.0f) * 0.5f;
        }
        __syncthreads();
    }
}

/*
 * glyph_match_top_k - Extract top-k glyph matches
 *
 * Grid: (num_patches, 1, 1)
 * Block: (256, 1, 1)
 *
 * Args:
 *   scores: Input scores [num_patches, num_glyphs]
 *   top_indices: Output top-k glyph indices [num_patches, k]
 *   top_scores: Output top-k scores [num_patches, k]
 *   num_patches: Number of patches
 *   num_glyphs: Number of glyph classes
 *   k: Number of top matches to extract (typically 3-5)
 */
extern "C" __global__ void glyph_match_top_k(
    const float* __restrict__ scores,
    int* __restrict__ top_indices,
    float* __restrict__ top_scores,
    int num_patches,
    int num_glyphs,
    int k
) {
    const int patch_idx = blockIdx.x;
    if (patch_idx >= num_patches) return;

    const int tid = threadIdx.x;

    // Shared memory for top-k selection
    __shared__ float local_scores[256];
    __shared__ int local_indices[256];

    // Load scores
    if (tid < num_glyphs) {
        local_scores[tid] = scores[patch_idx * num_glyphs + tid];
        local_indices[tid] = tid;
    } else {
        local_scores[tid] = -1.0f;
        local_indices[tid] = -1;
    }
    __syncthreads();

    // Parallel selection sort for top-k (simple but effective for small k)
    for (int rank = 0; rank < k; ++rank) {
        // Find maximum in remaining elements
        float max_score = -2.0f;
        int max_idx = -1;

        for (int i = tid; i < num_glyphs; i += blockDim.x) {
            if (local_scores[i] > max_score) {
                max_score = local_scores[i];
                max_idx = i;
            }
        }

        // Warp-level reduction to find global max
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_score = __shfl_down_sync(0xffffffff, max_score, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_score > max_score) {
                max_score = other_score;
                max_idx = other_idx;
            }
        }

        // Thread 0 writes result
        if (tid == 0 && max_idx >= 0) {
            top_scores[patch_idx * k + rank] = max_score;
            top_indices[patch_idx * k + rank] = local_indices[max_idx];
            local_scores[max_idx] = -2.0f;  // Mark as used
        }
        __syncthreads();
    }
}
