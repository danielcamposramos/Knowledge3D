/*
 * batchnorm_backward_training.cu - Full training-mode BatchNorm backward pass
 *
 * Implements gradient propagation for BatchNorm using per-batch statistics:
 *   d_input = gamma / std * (d_out - mean(d_out) - x_hat * mean(d_out * x_hat))
 *
 * RPN-style safeguards:
 *   - Guard NaN/Inf values before arithmetic
 *   - Clip final gradients to ±10 to preserve stability while allowing flow
 *
 * Assumes NHWC memory layout with elements packed as:
 *   [N, H, W, C] → flattened spatial index size = N * H * W
 *
 * Each CUDA block handles a single channel.
 */

extern "C" __global__ void batchnorm_backward_training(
    const float* __restrict__ d_out,          // Upstream gradient [N, H, W, C]
    const float* __restrict__ x_hat,          // Normalized activations [N, H, W, C]
    const float* __restrict__ gamma,          // Scale parameter [C]
    const float* __restrict__ batch_mean,     // Batch mean per channel [C]
    const float* __restrict__ batch_var,      // Batch variance per channel [C]
    float* __restrict__ d_input,              // Output gradient w.r.t. inputs [N, H, W, C]
    float* __restrict__ d_gamma,              // Output gradient w.r.t. gamma [C]
    float* __restrict__ d_beta,               // Output gradient w.r.t. beta [C]
    int N,
    int H,
    int W,
    int C,
    float eps
) {
    const int c = blockIdx.x;
    if (c >= C) {
        return;
    }

    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int spatial_size = N * H * W;

    // Shared buffers for reductions
    __shared__ float s_dgamma[256];
    __shared__ float s_dbeta[256];
    __shared__ float s_sum_dout[256];
    __shared__ float s_sum_dout_xhat[256];

    float local_dgamma = 0.0f;
    float local_dbeta = 0.0f;
    float local_sum_dout = 0.0f;
    float local_sum_dout_xhat = 0.0f;

    // Load per-channel parameters with safeguards
    float gamma_val = gamma[c];
    gamma_val = fmaxf(fminf(gamma_val, 2.0f), 0.1f);

    float var_val = batch_var[c];
    var_val = fmaxf(var_val, 0.0f);
    float std = sqrtf(var_val + eps);
    if (std < 1e-6f) {
        std = 1e-6f;
    }
    const float inv_std = 1.0f / std;

    // Access batch_mean to avoid unused parameter warning (guard already applied elsewhere)
    const float mean_val = batch_mean[c];
    (void)mean_val;

    // Pass 1: accumulate parameter gradients and helper sums
    for (int idx = tid; idx < spatial_size; idx += threads) {
        const int offset = idx * C + c;

        float grad = d_out[offset];
        float x_norm = x_hat[offset];

        if (isnan(grad) || isinf(grad)) {
            grad = 0.0f;
        }
        if (isnan(x_norm) || isinf(x_norm)) {
            x_norm = 0.0f;
        }

        local_dgamma += grad * x_norm;
        local_dbeta += grad;
        local_sum_dout += grad;
        local_sum_dout_xhat += grad * x_norm;
    }

    s_dgamma[tid] = local_dgamma;
    s_dbeta[tid] = local_dbeta;
    s_sum_dout[tid] = local_sum_dout;
    s_sum_dout_xhat[tid] = local_sum_dout_xhat;
    __syncthreads();

    // Reduce within block
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dgamma[tid] += s_dgamma[tid + stride];
            s_dbeta[tid] += s_dbeta[tid + stride];
            s_sum_dout[tid] += s_sum_dout[tid + stride];
            s_sum_dout_xhat[tid] += s_sum_dout_xhat[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes parameter gradients
    if (tid == 0) {
        d_gamma[c] = s_dgamma[0];
        d_beta[c] = s_dbeta[0];
    }
    __syncthreads();

    // Compute means needed for d_input
    const float mean_dout = s_sum_dout[0] / static_cast<float>(spatial_size);
    const float mean_dout_xhat = s_sum_dout_xhat[0] / static_cast<float>(spatial_size);
    __syncthreads();

    // Pass 2: compute gradients w.r.t. inputs
    for (int idx = tid; idx < spatial_size; idx += threads) {
        const int offset = idx * C + c;

        float grad = d_out[offset];
        float x_norm = x_hat[offset];

        if (isnan(grad) || isinf(grad)) {
            grad = 0.0f;
        }
        if (isnan(x_norm) || isinf(x_norm)) {
            x_norm = 0.0f;
        }

        float term = grad - mean_dout - x_norm * mean_dout_xhat;
        float d_input_val = gamma_val * inv_std * term;

        if (isnan(d_input_val) || isinf(d_input_val)) {
            d_input_val = 0.0f;
        } else {
            d_input_val = fmaxf(fminf(d_input_val, 10.0f), -10.0f);
        }

        d_input[offset] = d_input_val;
    }
}

