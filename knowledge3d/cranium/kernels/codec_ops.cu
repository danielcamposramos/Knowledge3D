// Ternary codec operations (quantize/dequantize) for sovereign pipelines.
//
// Quantisation maps float -> {-1, 0, +1} with a threshold.
// Dequantisation maps {-1,0,+1} -> float using unit steps.

#include <cuda_runtime.h>

__device__ __forceinline__ int ternary_quant_scalar(float v, float threshold) {
    if (v > threshold) return 1;
    if (v < -threshold) return -1;
    return 0;
}

extern "C" __global__
void ternary_quant_kernel(const float* input,
                          int* output,
                          int length,
                          float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    output[idx] = ternary_quant_scalar(input[idx], threshold);
}

extern "C" __global__
void ternary_dequant_kernel(const int* input,
                            float* output,
                            int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    int v = input[idx];
    // Clamp to {-1,0,1} just in case
    if (v > 1) v = 1;
    if (v < -1) v = -1;
    output[idx] = static_cast<float>(v);
}

// ------------------------------------------------------------------------- //
// 8x8 DCT kernels (orthonormal scaling) for batch of contiguous blocks.
// Input/Output layout: block-major, each block 64 floats in row-major order.
// One CUDA block (64 threads) processes one 8x8 block.
// ------------------------------------------------------------------------- //

__device__ __forceinline__ float _cosf_fast(float x) { return __cosf(x); }

extern "C" __global__
void dct8x8_forward_blocks(const float* input, int num_blocks, float* output) {
    int b = blockIdx.x;
    if (b >= num_blocks) return;
    int tid = threadIdx.x;  // 0..63
    int u = tid / 8;
    int v = tid % 8;

    const float PI = 3.14159265358979f;
    float alpha_u = (u == 0) ? rsqrtf(2.0f) : 1.0f;
    float alpha_v = (v == 0) ? rsqrtf(2.0f) : 1.0f;
    float sum = 0.0f;
    const float* block = input + b * 64;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float val = block[y * 8 + x];
            float cu = _cosf_fast((PI / 8.0f) * (x + 0.5f) * u);
            float cv = _cosf_fast((PI / 8.0f) * (y + 0.5f) * v);
            sum = fmaf(val, cu * cv, sum);
        }
    }
    output[b * 64 + tid] = 0.25f * alpha_u * alpha_v * sum;
}

extern "C" __global__
void dct8x8_inverse_blocks(const float* coeffs, int num_blocks, float* output) {
    int b = blockIdx.x;
    if (b >= num_blocks) return;
    int tid = threadIdx.x;  // 0..63
    int y = tid / 8;
    int x = tid % 8;
    const float PI = 3.14159265358979f;
    float sum = 0.0f;
    const float* block = coeffs + b * 64;
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            float c = block[u * 8 + v];
            float alpha_u = (u == 0) ? rsqrtf(2.0f) : 1.0f;
            float alpha_v = (v == 0) ? rsqrtf(2.0f) : 1.0f;
            float cu = _cosf_fast((PI / 8.0f) * (x + 0.5f) * u);
            float cv = _cosf_fast((PI / 8.0f) * (y + 0.5f) * v);
            sum = fmaf(alpha_u * alpha_v * c, cu * cv, sum);
        }
    }
    output[b * 64 + tid] = 0.25f * sum;
}

// ------------------------------------------------------------------------- //
// MDCT/IMDCT placeholder kernels (identity copy) to keep GPU path sovereign.
// Input/Output layout: contiguous frames of length frame_size.
// ------------------------------------------------------------------------- //

extern "C" __global__
void mdct_frame_identity(const float* input, float* output, int frame_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frame_size) return;
    output[idx] = input[idx];
}

extern "C" __global__
void imdct_frame_identity(const float* input, float* output, int frame_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frame_size) return;
    output[idx] = input[idx];
}

// ------------------------------------------------------------------------- //
// Real MDCT/IMDCT kernels.
// input:  frame_size samples (windowed)
// mdct:   frame_size / 2 coefficients
// imdct:  frame_size samples reconstructed from frame_size / 2 coeffs
// ------------------------------------------------------------------------- //

__device__ __forceinline__ float _cos_mdct(float x) {
    return __cosf(x);
}

extern "C" __global__
void mdct_forward_kernel(const float* input, float* output, int frame_size) {
    extern __shared__ float s_frame[];

    const int tid = threadIdx.x;
    const int N = frame_size;
    const int half_N = N >> 1;
    const float pi_over_N = (2.0f * 3.14159265358979323846f) / static_cast<float>(N);
    const float time_shift = 0.26f * static_cast<float>(N);

    // Stage full frame into shared memory for reuse across k threads
    for (int n = tid; n < N; n += blockDim.x) {
        s_frame[n] = input[n];
    }
    __syncthreads();

    const int k = blockIdx.x * blockDim.x + tid;
    if (k >= half_N) {
        return;
    }

    // Phase-aligned MDCT
    const float shift = time_shift;
    const float k_term = static_cast<float>(k) + 0.5f;

    float sum = 0.0f;
#pragma unroll 4
    for (int n = 0; n < N; ++n) {
        const float angle = pi_over_N * (static_cast<float>(n) + 0.5f + shift) * k_term;
        sum = fmaf(s_frame[n], _cos_mdct(angle), sum);
    }

    output[k] = sum;
}

extern "C" __global__
void imdct_inverse_kernel(const float* input, float* output, int frame_size) {
    extern __shared__ float s_coeffs[];

    const int tid = threadIdx.x;
    const int N = frame_size;
    const int half_N = N >> 1;
    const float pi_over_N = (2.0f * 3.14159265358979323846f) / static_cast<float>(N);
    const float time_shift = 0.26f * static_cast<float>(N);

    // Stage coefficients into shared memory
    for (int k = tid; k < half_N; k += blockDim.x) {
        s_coeffs[k] = input[k];
    }
    __syncthreads();

    const int n = blockIdx.x * blockDim.x + tid;
    if (n >= N) {
        return;
    }

    const float shift = time_shift;
    const float n_term = static_cast<float>(n) + 0.5f + shift;
    const float scale = 2.0f / static_cast<float>(N);

    float sum = 0.0f;
#pragma unroll 4
    for (int k = 0; k < half_N; ++k) {
        const float angle = pi_over_N * n_term * (static_cast<float>(k) + 0.5f);
        sum = fmaf(s_coeffs[k], _cos_mdct(angle), sum);
    }

    // Match numpy.hanning synthesis window to suppress edge artefacts
    const float window = 0.5f - 0.5f * _cos_mdct((2.0f * 3.14159265358979323846f * static_cast<float>(n)) /
                                                static_cast<float>(N - 1));
    output[n] = sum * scale * window;
}

// ------------------------------------------------------------------------- //
// Block layout transforms.
// These make RESHAPE_TO_BLOCKS / BLOCKS_TO_GRID first-class executable ops
// instead of vocabulary-only tokens.
// ------------------------------------------------------------------------- //

extern "C" __global__
void reshape_to_blocks_f32_kernel(
    const float* input,
    float* output,
    int rows,
    int cols,
    int block_h,
    int block_w
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) {
        return;
    }

    const int row = idx / cols;
    const int col = idx % cols;
    const int blocks_per_row = cols / block_w;
    const int block_area = block_h * block_w;
    const int block_row = row / block_h;
    const int block_col = col / block_w;
    const int local_row = row % block_h;
    const int local_col = col % block_w;
    const int block_index = block_row * blocks_per_row + block_col;
    const int local_index = local_row * block_w + local_col;

    output[block_index * block_area + local_index] = input[idx];
}

extern "C" __global__
void blocks_to_grid_f32_kernel(
    const float* input,
    float* output,
    int rows,
    int cols,
    int block_h,
    int block_w
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) {
        return;
    }

    const int row = idx / cols;
    const int col = idx % cols;
    const int blocks_per_row = cols / block_w;
    const int block_area = block_h * block_w;
    const int block_row = row / block_h;
    const int block_col = col / block_w;
    const int local_row = row % block_h;
    const int local_col = col % block_w;
    const int block_index = block_row * blocks_per_row + block_col;
    const int local_index = local_row * block_w + local_col;

    output[idx] = input[block_index * block_area + local_index];
}

extern "C" __global__
void reshape_to_blocks_i32_kernel(
    const int* input,
    int* output,
    int rows,
    int cols,
    int block_h,
    int block_w
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) {
        return;
    }

    const int row = idx / cols;
    const int col = idx % cols;
    const int blocks_per_row = cols / block_w;
    const int block_area = block_h * block_w;
    const int block_row = row / block_h;
    const int block_col = col / block_w;
    const int local_row = row % block_h;
    const int local_col = col % block_w;
    const int block_index = block_row * blocks_per_row + block_col;
    const int local_index = local_row * block_w + local_col;

    output[block_index * block_area + local_index] = input[idx];
}

extern "C" __global__
void blocks_to_grid_i32_kernel(
    const int* input,
    int* output,
    int rows,
    int cols,
    int block_h,
    int block_w
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) {
        return;
    }

    const int row = idx / cols;
    const int col = idx % cols;
    const int blocks_per_row = cols / block_w;
    const int block_area = block_h * block_w;
    const int block_row = row / block_h;
    const int block_col = col / block_w;
    const int local_row = row % block_h;
    const int local_col = col % block_w;
    const int block_index = block_row * blocks_per_row + block_col;
    const int local_index = local_row * block_w + local_col;

    output[idx] = input[block_index * block_area + local_index];
}
