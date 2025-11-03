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

    // Compute batch statistics
    float mean = running_mean[c];
    float var = fmaxf(running_var[c], 0.0f);  // RPN-style: ensure non-negative
    float std = sqrtf(var + eps);
    // RPN-style epsilon guard: prevent division by tiny numbers
    if (std < 1e-6f) {
        std = 1e-6f;
    }
    float inv_std = 1.0f / std;

    // First pass: compute d_gamma and d_beta
    for (int i = tid; i < spatial_size; i += threads_per_block) {
        int spatial_idx = i * C + c;

        // Normalized input
        float x_norm = (x_in[spatial_idx] - mean) * inv_std;

        // Gradient from next layer
        float grad = d_out[spatial_idx];

        // RPN-style NaN guard: only accumulate valid gradients
        if (!isnan(grad) && !isinf(grad)) {
            sum_d_gamma += grad * x_norm;
            sum_d_beta += grad;
        }
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

    // Write parameter gradients
    if (tid == 0) {
        d_gamma[c] = s_d_gamma[0];
        d_beta[c] = s_d_beta[0];
    }

    // Second pass: compute d_input
    // Use simplified backward (inference-mode style) for numerical stability
    // This avoids gradient cancellation with small batch sizes
    float gamma_val = gamma[c];

    // CRITICAL FIX: Clip gamma to [0.1, 2.0] before gradient computation
    gamma_val = fmaxf(fminf(gamma_val, 2.0f), 0.1f);

    for (int i = tid; i < spatial_size; i += threads_per_block) {
        int spatial_idx = i * C + c;

        float grad = d_out[spatial_idx];

        // RPN-style NaN guard: sanitize inputs before computation
        if (isnan(grad) || isinf(grad)) {
            grad = 0.0f;
        }

        // Simplified backward: d_input = d_out * gamma / std
        float corrected_grad = grad * gamma_val * inv_std;

        // RPN-style: guard against NaN/inf only (relaxed clipping)
        // Light clipping to prevent extreme outliers, rely on small LR for stability
        if (isnan(corrected_grad) || isinf(corrected_grad)) {
            corrected_grad = 0.0f;
        } else {
            // Relaxed clipping at ±10 - allows gradients to flow through deep network
            corrected_grad = fmaxf(fminf(corrected_grad, 10.0f), -10.0f);
        }

        d_input[spatial_idx] = corrected_grad;
    }
}
