/*
 * trigram_embed.cu - GPU trigram embedding lookup and normalization
 *
 * Enables RPN trigram embeddings to stay GPU-sovereign by performing
 * lookup, averaging, and optional L2 normalization entirely on device.
 * All arithmetic follows the sovereign guard pattern (NaN/Inf sanitation
 * plus relaxed ±10 clipping).
 */

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void trigram_lookup_average(
    const int* __restrict__ trigram_indices,  // [num_trigrams]
    const float* __restrict__ embedding_table, // [vocab_size, embed_dim]
    float* __restrict__ output,               // [embed_dim]
    int num_trigrams,
    int embed_dim,
    int vocab_size
) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= embed_dim) {
        return;
    }

    float sum = 0.0f;
    for (int t = 0; t < num_trigrams; ++t) {
        int idx = trigram_indices[t];
        if (idx < 0 || idx >= vocab_size) {
            continue;
        }
        float val = embedding_table[idx * embed_dim + dim];
        if (!isnan(val) && !isinf(val)) {
            sum += val;
        }
    }

    float avg = (num_trigrams > 0) ? (sum / static_cast<float>(num_trigrams)) : 0.0f;
    if (isnan(avg) || isinf(avg)) {
        avg = 0.0f;
    }
    avg = fmaxf(fminf(avg, 10.0f), -10.0f);
    output[dim] = avg;
}

extern "C" __global__ void l2_normalize_embedding(
    float* __restrict__ embedding,
    int embed_dim
) {
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;
    float local = 0.0f;

    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float v = embedding[i];
        if (isnan(v) || isinf(v)) {
            v = 0.0f;
        }
        local += v * v;
    }

    shared_sum[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    float norm = sqrtf(shared_sum[0] + 1e-8f);
    float inv = 1.0f / norm;

    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float v = embedding[i];
        v = fmaxf(fminf(v, 10.0f), -10.0f);
        embedding[i] = v * inv;
    }
}
