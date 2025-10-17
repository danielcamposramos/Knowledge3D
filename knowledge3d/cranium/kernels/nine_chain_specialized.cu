/**
 * Step 14 – Specialized Nine-Chain Swarm Kernels
 *
 * This file contains the production-ready CUDA kernels for the 9-chain
 * bio-inspired swarm. Each chain implements a distinct reasoning style,
 * and the resonance kernel produces inter-chain similarity scores with
 * shared-memory reductions. Chain vectors are assumed to have length 128
 * (CHAIN_DIM), matching the ThinkingTag SPATIAL embedding size.
 */

#include <cuda_runtime.h>
#include <math_constants.h>

constexpr int CHAIN_DIM = 128;
constexpr int NUM_ACTIVE_CHAINS = 8;  // chains 1-8 feed synthesis

// Helpers ------------------------------------------------------------------

__device__ inline float device_sqrt(float x) {
    return sqrtf(fmaxf(x, 0.0f));
}

__device__ inline float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ inline float fast_tanh(float x) {
    return tanhf(x);
}

// Chain 1: Ingest -----------------------------------------------------------

__device__ inline void chain_ingest_impl(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    __shared__ float s_input[CHAIN_DIM];

    int tid = threadIdx.x;
    if (tid < dim) {
        s_input[tid] = input[tid];
    }
    __syncthreads();

    if (tid < dim) {
        // 8-sample local window statistics (wrap around for simplicity)
        const int window = 8;
        float mean = 0.0f;
        for (int i = 0; i < window; ++i) {
            int idx = (tid * window + i) % dim;
            mean += s_input[idx];
        }
        mean /= window;

        float variance = 0.0f;
        for (int i = 0; i < window; ++i) {
            int idx = (tid * window + i) % dim;
            float diff = s_input[idx] - mean;
            variance += diff * diff;
        }
        variance /= window;

        float signal = 0.5f * s_input[tid] + 0.3f * mean + 0.2f * device_sqrt(variance);
        output[tid] = signal;
        state[tid] = signal;
    }
}

extern "C" __global__ void chain_ingest_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_ingest_impl(input, output, state, dim);
}

// Chain 2: Fuse-A (associative / semantic) ---------------------------------

__device__ inline void chain_fuse_a_impl(
    const float* __restrict__ ingest_output,
    const float* __restrict__ prev_state,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    __shared__ float s_ingest[CHAIN_DIM];
    __shared__ float s_prev[CHAIN_DIM];

    int tid = threadIdx.x;
    if (tid < dim) {
        s_ingest[tid] = ingest_output[tid];
        s_prev[tid] = prev_state ? prev_state[tid] : 0.0f;
    }
    __syncthreads();

    if (tid < dim) {
        // Attention-like scalar comparing ingest and previous state
        float attention = 0.0f;
        for (int i = 0; i < dim; ++i) {
            attention += s_ingest[tid] * s_ingest[i];
        }
        attention = fast_tanh(attention / (float)dim);

        float fused = 0.6f * s_ingest[tid] + 0.3f * attention * s_ingest[tid] + 0.1f * s_prev[tid];
        output[tid] = fused;
        state[tid] = fused;
    }
}

extern "C" __global__ void chain_fuse_a_kernel(
    const float* __restrict__ ingest_output,
    const float* __restrict__ prev_state,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_fuse_a_impl(ingest_output, prev_state, output, state, dim);
}

// Chain 3: Fuse-B (logical / structural) -----------------------------------

__device__ inline void chain_fuse_b_impl(
    const float* __restrict__ ingest_output,
    const float* __restrict__ fuse_a_output,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    __shared__ float s_ingest[CHAIN_DIM];
    __shared__ float s_fuse_a[CHAIN_DIM];

    int tid = threadIdx.x;
    if (tid < dim) {
        s_ingest[tid] = ingest_output[tid];
        s_fuse_a[tid] = fuse_a_output[tid];
    }
    __syncthreads();

    if (tid < dim) {
        // Structural contrast with neighbour segment
        const int segment = 4;
        float logical_signal = 0.0f;
        for (int i = 0; i < segment; ++i) {
            int idx1 = (tid + i) % dim;
            int idx2 = (tid + i + segment) % dim;
            logical_signal += fabsf(s_ingest[idx1] - s_ingest[idx2]);
        }
        logical_signal /= segment;

        float fused = s_fuse_a[tid] * (1.0f + 0.25f * fast_tanh(logical_signal));
        output[tid] = fused;
        state[tid] = fused;
    }
}

extern "C" __global__ void chain_fuse_b_kernel(
    const float* __restrict__ ingest_output,
    const float* __restrict__ fuse_a_output,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_fuse_b_impl(ingest_output, fuse_a_output, output, state, dim);
}

// Chain 4: Spatial-A (geometric) -------------------------------------------

__device__ inline void chain_spatial_a_impl(
    const float* __restrict__ fuse_b_output,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    __shared__ float s_input[CHAIN_DIM];

    int tid = threadIdx.x;
    if (tid < dim) {
        s_input[tid] = fuse_b_output[tid];
    }
    __syncthreads();

    if (tid < dim) {
        // Treat vector as 8x16 grid
        const int rows = 8;
        const int cols = 16;
        int row = tid / cols;
        int col = tid % cols;

        float center = s_input[tid];
        float right = (col + 1 < cols) ? s_input[row * cols + (col + 1)] : center;
        float down = (row + 1 < rows) ? s_input[(row + 1) * cols + col] : center;

        float dx = right - center;
        float dy = down - center;
        float gradient = device_sqrt(dx * dx + dy * dy);

        float enhanced = center + 0.35f * gradient;
        output[tid] = enhanced;
        state[tid] = enhanced;
    }
}

extern "C" __global__ void chain_spatial_a_kernel(
    const float* __restrict__ fuse_b_output,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_spatial_a_impl(fuse_b_output, output, state, dim);
}

// Chain 5: Spatial-B (topological) -----------------------------------------

__device__ inline void chain_spatial_b_impl(
    const float* __restrict__ fuse_b_output,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    __shared__ float s_input[CHAIN_DIM];

    int tid = threadIdx.x;
    if (tid < dim) {
        s_input[tid] = fuse_b_output[tid];
    }
    __syncthreads();

    if (tid < dim) {
        const int neighbourhood = 5;
        float density = 0.0f;
        for (int offset = -neighbourhood; offset <= neighbourhood; ++offset) {
            int idx = tid + offset;
            if (idx >= 0 && idx < dim) {
                density += s_input[idx] > 0.4f ? 1.0f : 0.0f;
            }
        }
        density /= (2 * neighbourhood + 1);

        float enhanced = s_input[tid] * (0.7f + 0.3f * density);
        output[tid] = enhanced;
        state[tid] = enhanced;
    }
}

extern "C" __global__ void chain_spatial_b_kernel(
    const float* __restrict__ fuse_b_output,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_spatial_b_impl(fuse_b_output, output, state, dim);
}

// Chain 6: Spatial-C (temporal-spatial) ------------------------------------

__device__ inline void chain_spatial_c_impl(
    const float* __restrict__ fuse_b_output,
    const float* __restrict__ prev_state,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    int tid = threadIdx.x;
    if (tid < dim) {
        float current = fuse_b_output[tid];
        float previous = prev_state ? prev_state[tid] : current;
        float derivative = current - previous;

        float prev_derivative = 0.0f;
        if (tid > 0) {
            float prev_curr = fuse_b_output[tid - 1];
            float prev_prev = prev_state ? prev_state[tid - 1] : prev_curr;
            prev_derivative = prev_curr - prev_prev;
        }
        float acceleration = derivative - prev_derivative;

        float temporal_feature = current + 0.25f * derivative + 0.1f * acceleration;
        output[tid] = temporal_feature;
        state[tid] = temporal_feature;
    }
}

extern "C" __global__ void chain_spatial_c_kernel(
    const float* __restrict__ fuse_b_output,
    const float* __restrict__ prev_state,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_spatial_c_impl(fuse_b_output, prev_state, output, state, dim);
}

// Chain 7: Reason (reductionist) -------------------------------------------

__device__ inline void chain_reason_reductionist_impl(
    const float* __restrict__ spatial_a,
    const float* __restrict__ spatial_b,
    const float* __restrict__ spatial_c,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    int tid = threadIdx.x;
    if (tid < dim) {
        float analytical = 0.55f * spatial_a[tid]
                         + 0.25f * spatial_b[tid]
                         + 0.20f * spatial_c[tid];
        analytical = fast_tanh(analytical);

        output[tid] = analytical;
        state[tid] = analytical;
    }
}

extern "C" __global__ void chain_reason_reductionist_kernel(
    const float* __restrict__ spatial_a,
    const float* __restrict__ spatial_b,
    const float* __restrict__ spatial_c,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_reason_reductionist_impl(spatial_a, spatial_b, spatial_c, output, state, dim);
}

// Chain 8: Reason (creative) ------------------------------------------------

__device__ inline void chain_reason_creative_impl(
    const float* __restrict__ spatial_a,
    const float* __restrict__ spatial_b,
    const float* __restrict__ spatial_c,
    const float* __restrict__ fuse_a,
    const float* __restrict__ fuse_b,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    int tid = threadIdx.x;
    if (tid < dim) {
        float creative_mix = spatial_a[tid] * fuse_a[tid]
                           + spatial_b[tid] * fuse_b[tid]
                           + spatial_c[tid] * 0.3f;

        creative_mix += 0.35f * sinf(creative_mix * CUDART_PI_F);
        creative_mix = fast_tanh(creative_mix);

        output[tid] = creative_mix;
        state[tid] = creative_mix;
    }
}

extern "C" __global__ void chain_reason_creative_kernel(
    const float* __restrict__ spatial_a,
    const float* __restrict__ spatial_b,
    const float* __restrict__ spatial_c,
    const float* __restrict__ fuse_a,
    const float* __restrict__ fuse_b,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_reason_creative_impl(spatial_a, spatial_b, spatial_c, fuse_a, fuse_b, output, state, dim);
}

// Resonance computation -----------------------------------------------------

extern "C" __global__ void compute_resonance_optimized(
    const float* __restrict__ chain_outputs,
    float* __restrict__ resonance_matrix,
    int dim) {
    int chain_a = blockIdx.x;
    int chain_b = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float partial[256];

    float sum = 0.0f;
    for (int idx = tid; idx < dim; idx += blockDim.x) {
        float a = chain_outputs[chain_a * dim + idx];
        float b = chain_outputs[chain_b * dim + idx];
        sum += a * b;
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float norm = (float)dim;
        resonance_matrix[chain_a * NUM_ACTIVE_CHAINS + chain_b] = partial[0] / norm;
    }
}

__device__ inline void compute_resonance_matrix_serial(
    const float* __restrict__ chain_outputs,
    float* __restrict__ resonance_matrix,
    int dim) {
    __shared__ float partial[256];
    int tid = threadIdx.x;

    for (int chain_a = 0; chain_a < NUM_ACTIVE_CHAINS; ++chain_a) {
        for (int chain_b = chain_a; chain_b < NUM_ACTIVE_CHAINS; ++chain_b) {
            float sum = 0.0f;
            for (int idx = tid; idx < dim; idx += blockDim.x) {
                float a = chain_outputs[chain_a * dim + idx];
                float b = chain_outputs[chain_b * dim + idx];
                sum += a * b;
            }
            partial[tid] = sum;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial[tid] += partial[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                float value = partial[0] / (float)dim;
                resonance_matrix[chain_a * NUM_ACTIVE_CHAINS + chain_b] = value;
                if (chain_b != chain_a) {
                    resonance_matrix[chain_b * NUM_ACTIVE_CHAINS + chain_a] = value;
                }
            }
            __syncthreads();
        }
    }
}

// Chain 9: Synthesis --------------------------------------------------------

__device__ inline void chain_synthesis_impl(
    const float* __restrict__ chain_outputs,
    const float* __restrict__ resonance_matrix,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    __shared__ float weights[NUM_ACTIVE_CHAINS];

    int tid = threadIdx.x;
    if (tid < NUM_ACTIVE_CHAINS) {
        const float* row = resonance_matrix + tid * NUM_ACTIVE_CHAINS;
        float accum = 0.0f;
        for (int j = 0; j < NUM_ACTIVE_CHAINS; ++j) {
            accum += fabsf(row[j]);
        }
        weights[tid] = accum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int c = 0; c < NUM_ACTIVE_CHAINS; ++c) {
            total += weights[c];
        }
        if (total < 1e-6f) {
            total = 1e-6f;
        }
        for (int c = 0; c < NUM_ACTIVE_CHAINS; ++c) {
            weights[c] /= total;
        }
    }
    __syncthreads();

    if (tid < dim) {
        float weighted_sum = 0.0f;
        for (int chain = 0; chain < NUM_ACTIVE_CHAINS; ++chain) {
            weighted_sum += weights[chain] * chain_outputs[chain * dim + tid];
        }
        output[tid] = weighted_sum;
        state[tid] = weighted_sum;
    }
}

extern "C" __global__ void chain_synthesis_kernel(
    const float* __restrict__ chain_outputs,
    const float* __restrict__ resonance_matrix,
    float* __restrict__ output,
    float* __restrict__ state,
    int dim) {
    chain_synthesis_impl(chain_outputs, resonance_matrix, output, state, dim);
}

extern "C" __global__ void swarm_iteration_kernel(
    const float* __restrict__ input,
    float* __restrict__ chain_buffers,
    float* __restrict__ chain9_state,
    float* __restrict__ resonance_matrix,
    int dim) {
    float* chain0 = chain_buffers + 0 * dim;
    float* chain1 = chain_buffers + 1 * dim;
    float* chain2 = chain_buffers + 2 * dim;
    float* chain3 = chain_buffers + 3 * dim;
    float* chain4 = chain_buffers + 4 * dim;
    float* chain5 = chain_buffers + 5 * dim;
    float* chain6 = chain_buffers + 6 * dim;
    float* chain7 = chain_buffers + 7 * dim;

    chain_ingest_impl(input, chain0, chain0, dim);
    __syncthreads();

    chain_fuse_a_impl(chain0, chain1, chain1, chain1, dim);
    __syncthreads();

    chain_fuse_b_impl(chain0, chain1, chain2, chain2, dim);
    __syncthreads();

    chain_spatial_a_impl(chain2, chain3, chain3, dim);
    __syncthreads();

    chain_spatial_b_impl(chain2, chain4, chain4, dim);
    __syncthreads();

    chain_spatial_c_impl(chain2, chain5, chain5, chain5, dim);
    __syncthreads();

    chain_reason_reductionist_impl(chain3, chain4, chain5, chain6, chain6, dim);
    __syncthreads();

    chain_reason_creative_impl(chain3, chain4, chain5, chain1, chain2, chain7, chain7, dim);
    __syncthreads();

    compute_resonance_matrix_serial(chain_buffers, resonance_matrix, dim);
    __syncthreads();

    chain_synthesis_impl(chain_buffers, resonance_matrix, chain9_state, chain9_state, dim);
}
