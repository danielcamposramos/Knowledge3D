/*
 * batchnorm_backward.cu - BatchNorm Backward Pass
 *
 * Computes gradients for BatchNorm layer:
 *   - d_gamma: gradient w.r.t. scale parameter
 *   - d_beta: gradient w.r.t. shift parameter
 *   - d_input: gradient w.r.t. input
 *
 * Uses running statistics (inference mode) for simplicity.
 * Each thread block handles one channel.
 */

extern "C" __global__ void batchnorm_backward(
    const float* __restrict__ d_out,          // Gradient from next layer [H, W, C]
    const float* __restrict__ x_in,           // Input before normalization [H, W, C]
    const float* __restrict__ gamma,          // Scale [C]
    const float* __restrict__ running_mean,   // Running mean [C]
    const float* __restrict__ running_var,    // Running variance [C]
    float* __restrict__ d_input,              // Output: d_input [H, W, C]
    float* __restrict__ d_gamma,              // Output: d_gamma [C]
    float* __restrict__ d_beta,               // Output: d_beta [C]
    int H,
    int W,
    int C,
    float eps
) {
    int c = blockIdx.x;  // One block per channel
    if (c >= C) return;

    int tid = threadIdx.x;
    int spatial_size = H * W;
    int threads_per_block = blockDim.x;

    // Shared memory for reductions
    __shared__ float s_d_gamma[256];
    __shared__ float s_d_beta[256];

    float sum_d_gamma = 0.0f;
    float sum_d_beta = 0.0f;

    // Compute d_gamma and d_beta via reduction
    float mean = running_mean[c];
    float var = running_var[c];
    float std = sqrtf(var + eps);
    float inv_std = 1.0f / std;

    for (int i = tid; i < spatial_size; i += threads_per_block) {
        int spatial_idx = i * C + c;

        // Normalized input
        float x_norm = (x_in[spatial_idx] - mean) * inv_std;

        // Accumulate gradients
        float grad = d_out[spatial_idx];
        sum_d_gamma += grad * x_norm;
        sum_d_beta += grad;
    }

    s_d_gamma[tid] = sum_d_gamma;
    s_d_beta[tid] = sum_d_beta;
    __syncthreads();

    // Reduction within block
    for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_d_gamma[tid] += s_d_gamma[tid + stride];
            s_d_beta[tid] += s_d_beta[tid + stride];
        }
        __syncthreads();
    }

    // Write reduced gradients
    if (tid == 0) {
        d_gamma[c] = s_d_gamma[0];
        d_beta[c] = s_d_beta[0];
    }

    // Compute d_input
    float gamma_val = gamma[c];
    for (int i = tid; i < spatial_size; i += threads_per_block) {
        int spatial_idx = i * C + c;

        // d_input = d_out * gamma / std
        d_input[spatial_idx] = d_out[spatial_idx] * gamma_val * inv_std;
    }
}
